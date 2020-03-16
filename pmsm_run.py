# Keep systems of equations up to date
# Efficiency plot needs to be finished
# Magnet losses need to be completed
# Eff inputs and outputs should match up with ccblade and zappy
# need to add torque/speed curve and print out plot
# Future: Calculate winding factor from Lipo book, see chapter two
# Future: output phasor diagrams

# Make models of 20 hp motors roughly the size of a small personal quadcopter
    # Change current density based on liquid/air cooling
    # Material may matter, can do a cost / mass study
    # What voltage and current
    # What are geometry constraints: diameter, length
    # Need to implement air gap calculation 
    # RPM from NDARC

import numpy as np
from math import pi

from openmdao.api import Problem, IndepVarComp, ExplicitComponent, ExecComp
from openmdao.api import NewtonSolver, Group, DirectSolver, NonlinearRunOnce, LinearRunOnce, view_model, BalanceComp

from electromagnetics.em_group import EmGroup
from thermal.thermal_group import ThermalGroup
from sizing.size_group import SizeGroup



if __name__ == "__main__":
    p = Problem()
    model = p.model

    ind = model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])


    # -------------------------------------------------------------------------
    #                              
    # -------------------------------------------------------------------------
    ind.add_output('radius_motor', 0.078225, units='m', desc='Motor outer radius')            # Ref motor = 0.078225
    ind.add_output('rpm', 5400, units='rpm', desc='Rotation speed')  
    ind.add_output('I', 34.8, units='A', desc='RMS Current')
    ind.add_output('V', 385, units='V', desc='RMS voltage')  
    ind.add_output('stack_length', 0.0345, units='m', desc='Stack Length')              # Ref motor = 0.0345

    ind.add_output('n_turns', 12, desc='Number of wire turns')
    ind.add_output('n_slots', 24, desc='Number of Slots')
    ind.add_output('n_m', 20, desc='Number of magnets')

    ind.add_output('t_mag', .0044, units='m', desc='Radial magnet thickness')               # Ref motor = 0.0044
    ind.add_output('r_strand', 0.0001605, units='m', desc='28 AWG radius of one strand of litz wire')
    ind.add_output('n_strands', 41, desc='number of strands in hand for litz wire')
  
    ind.add_output('T_windings', 150, units='C', desc='operating temperature of windings')
    ind.add_output('T_mag', 100, units='C', desc='operating temperature of the magnets')

    # -------------------------------------------------------------------------
    #                        Material Properties and Constants
    # -------------------------------------------------------------------------
    ind.add_output('Hc_20', -1046, units='kA/m', desc='Intrinsic Coercivity at 20 degC')     
    ind.add_output('Br_20', 1.39, units='T', desc='remnance flux density at 20 degC')           
    ind.add_output('k_sat', 1, desc='Saturation factor of the magnetic circuit due to the main (linkage) magnetic flux')
    ind.add_output('K_h', 0.007379032537)
    ind.add_output('K_e', 9.2630137e6)
    ind.add_output('K_h_alpha', 1.152932586)
    ind.add_output('K_h_beta', 1.72240394)             
    ind.add_output('alpha', 1.27, desc='core loss constant') 
    ind.add_output('mu_a', 1.81e-5, units='(m**2)/s', desc='air dynamic viscosity')
    ind.add_output('muf_b', .3, desc='bearing friction coefficient')
    ind.add_output('D_b', .01, units='m', desc='bearing bore diameter')
    ind.add_output('F_b', 100, units='N', desc='bearing load')          #coupled with motor mass
    ind.add_output('rho_a', 1.225, units='kg/m**3', desc='air density')
    ind.add_output('H_c', 1353000, units='A/m', desc='Intrinsic Coercivity of magnets constant')           
    ind.add_output('B_r',  1.39, units='T', desc='Remanence magnetic flux')                
    ind.add_output('mu_o',  0.4*pi*10**-6, units='H/m', desc='permeability of free space')   
    ind.add_output('mu_r',  1.0, units='H/m', desc='relative magnetic permeability of ferromagnetic materials')
    ind.add_output('rho', 8110.2, units='kg/m**3', desc='Density of Hiperco-50')  
    ind.add_output('rho_mag', 7500, units='kg/m**3', desc='Density of Magnets')
    ind.add_output('rho_wire', 8940, units='kg/m**3', desc='Density of wire: Cu=8940')
    ind.add_output('resistivity_wire', 1.724e-8, units='ohm*m', desc='resisitivity of Cu at 20 degC') 
    ind.add_output('T_coeff_cu', 0.00393, desc='temperature coeff of copper')
    # ind.add_output('T_ref_wire', 20.0, units='C', desc='reference temperature at which winding resistivity is measured')
    ind.add_output('alpha_stein', 1.286, desc='Alpha coefficient for steinmetz, constant')
    ind.add_output('beta_stein', 1.76835, desc='Beta coefficient for steinmentz, dependent on freq')
    ind.add_output('k_stein', 0.0044, desc='k constant for steinmentz')
    ind.add_output('T_coef_rem_mag', -0.12, desc=' Temperature coefficient of the remnance flux density for N48H magnets')

    # -------------------------------------------------------------------------

    ind.add_output('b_ry', 3.0, units='T', desc='Rotor yoke flux density')              # FEA
    ind.add_output('b_sy', 2.4, units='T', desc='Stator yoke flux density')             # FEA
    ind.add_output('b_t', 3.0, units='T', desc='Tooth Flux Density')                    # FEA
    ind.add_output('B_pk', 2.4, units='T', desc='Peak flux density for Hiperco-50')     # FEA
    ind.add_output('k_wb', 0.58, desc='copper fill factor')                             # Ref motor = 0.58

    ind.add_output('l_slot_opening', .00325, units='m', desc='length of the slot opening')  # Ref motor = .00325
    ind.add_output('k', 0.94, desc='Stacking factor assumption')
    ind.add_output('gap', 0.0010, units='m', desc='Air gap distance, Need to calculate effective air gap, using Carters Coeff')


    model.add_subsystem('em_properties', EmGroup(), promotes_inputs=['T_coef_rem_mag', 'T_mag', 'I', 'rpm', 'mu_r', 'g_eq', 't_mag', 'Br_20',
                                                                     'gap', 'sta_ir', 'n_slots', 'l_slot_opening', 't_mag',
                                                                     'carters_coef', 'k_sat', 'mu_o', 'P_wire', 'P_steinmetz',
                                                                     'B_g', 'n_m', 'n_turns', 'stack_length', 'rot_or', 's_d', 'w_t', 'slot_area', 't_mag', 'w_slot'], 
                                                    promotes_outputs=['P_in', 'P_out', 'B_g', 
                                                                      'mech_angle', 't_1', 'carters_coef',
                                                                      'g_eq', 'g_eq_q', 'Br', 'Eff', 
                                                                      'Tq', 'rot_volume', 'stator_surface_current'])
    
    model.add_subsystem('thermal_properties', ThermalGroup(), promotes_inputs=[ 'B_pk', 'alpha_stein', 'beta_stein', 'k_stein', 'rpm', 
                                                                               'resistivity_wire', 'stack_length', 'n_slots', 'n_strands', 
                                                                               'n_m', 'mu_o', 'n_turns', 'T_coeff_cu', 'I', 'T_windings', 'r_strand', 'mu_r'],
                                                                               # 'K_h_alpha', 'K_h_beta', 'K_h', 'K_e', D_b', 'F_b', 'alpha', 'gap', 'k', 'mu_a', 'muf_b', 'n_m', 'rho_a', 'rot_ir', 'rot_or', 'rpm', 'stack_length'],
                                                              promotes_outputs=[
                                                                                # 'L_core','L_emag', 'L_ewir', 'L_airg', 'L_airf', 'L_bear','L_total',
                                                                                'A_cu', 'r_litz', 'P_steinmetz', 'P_dc', 'P_ac', 'P_wire', 'L_wire', 'R_dc', 'skin_depth', 'temp_resistivity', 'f_e'])

    model.add_subsystem('geometry', SizeGroup(), promotes_inputs=['gap', 'B_g', 'k', 'b_ry', 'n_m',
                                                                'b_sy', 'b_t', 'n_turns', 'I', 'k_wb',
                                                                'rho', 'radius_motor', 'n_slots', 'sta_ir', 'w_t', 'stack_length',
                                                                's_d', 'rot_or', 'rot_ir', 't_mag', 'rho_mag'],
                                                 promotes_outputs=['J', 'w_ry', 'w_sy', 'w_t', 'sta_ir', 'rot_ir', 's_d',
                                                                 'mag_mass', 'sta_mass', 'rot_mass', 'slot_area', 'w_slot'])

    bal = BalanceComp()
    bal.add_balance('rot_or_state', val=0.05, units='m')#, use_mult=True, mult_val=0.5)
    tgt = IndepVarComp(name='J_tgt', val=10.47, units='A/mm**2')
    model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['J_tgt'])
    model.add_subsystem(name='balance', subsys=bal)

    model.connect('J_tgt', 'balance.rhs:rot_or_state')
    model.connect('balance.rot_or_state', 'rot_or')
    model.connect('J', 'balance.lhs:rot_or_state')


    model.linear_solver = DirectSolver()

    model.nonlinear_solver = NewtonSolver()
    model.nonlinear_solver.options['maxiter'] = 100
    model.nonlinear_solver.options['iprint'] = 0

    p.setup()
    p.final_setup()
    #p.check_partials(compact_print=True)
    # p.model.list_outputs(implicit=True)
    # p.set_solver_print(2)
    # view_model(p)
    # quit()
    p.run_model()
    print('-----------GEOMETRY---------------')
    # print('Rotor Inner Diameter..............', 2 * p.get_val('rot_ir', units='mm'))
    print('Rotor Inner radius................',     p.get_val('rot_ir', units='mm'))

    # print('Rotor Yoke Thickness..............',  p.get_val('w_ry', units='mm'))
    # print('Magnet Thickness..................',  p.get_val('t_mag', units='mm'))

    # print('Rotor outer Diameter..............',  2 * p.get_val('rot_or', units='mm'))
    print('Rotor outer Radius................',      p.get_val('balance.rot_or_state', units='mm'))

    # print('Stator Inner Diameter.............', 2 *  p.get_val('sta_ir', units='mm'))
    print('Stator Inner Radius...............',      p.get_val('sta_ir', units='mm'))

    print('Slot Depth........................',  p.get_val('s_d', units='mm'))
    print('Stator Yoke Thickness.............',  p.get_val('w_sy', units='mm'))

    # print('Motor Outer Diameter..............', 2 *  p.get_val('mass.radius_motor', units='mm'))
    print('Motor Outer Radius................',      p.get_val('radius_motor', units='mm'))

    print('Tooth Width.......................',  p.get_val('w_t', units='mm'))
    print('Radius of litz wire ..............', p.get_val('r_litz', units='m'))
    print('Length of Windings................',  p.get_val('L_wire', units='m'))
    print('Slot Area.........................', p.get_val('slot_area', units='m**2'))
    print('Slot Width........................', p.get_val('w_slot'))
    print('Copper area in one slot...........', p.get_val('A_cu', units='m**2'))
    print('Copper Slot Fill Percentage.......', p.get_val('A_cu') / p.get_val('slot_area'))

    print('--------------MASS----------------')
    print('Mass of Stator....................',  p.get_val('sta_mass', units='kg'))
    print('Mass of Rotor.....................',  p.get_val('rot_mass', units='kg'))
    print('Mass of Magnets...................',  p.get_val('mag_mass', units='kg')) 
    print('Mass of Motor.....................',  p.get_val('mag_mass', units='kg') + p.get_val('rot_mass', units='kg') + p.get_val('sta_mass', units='kg'))   
    

    print('--------------LOSSES-------------')
    print('Current Density.........',   p.get_val('J'))
    print('Iron losses.............',   p.get_val('P_steinmetz') * p.get_val('sta_mass'))
    print('DC Winding  Losses......',   p.get_val('P_dc'))
    print('AC Winding  Losses......',   p.get_val('P_ac'))
    print('TOTAL Winding  Losses...',   p.get_val('P_wire'))
    print('Total Losses............',   p.get_val('P_steinmetz') * p.get_val('sta_mass') + p.get_val('P_wire'))
    print('Overall Efficiency......',   p.get_val('Eff'))
    # print('Skin Depth..............',   p.get_val('skin_depth', units='mm'))
    # print('Temp Dependent Resistivity.......', p.get_val('temp_resistivity', units='ohm*m'))

    print('--------------EM PERF-------------')
    print('Power In  ........................',  p.get_val('P_in'))
    print('Power out  .......................',  p.get_val('P_out'))
    print('Electrical Frequency..............',  p.get_val('f_e'))
    print('Torque............................',  p['Tq'])

    print('--------------FIELDS--------------')
    print('Air gap flux density .............',  p.get_val('B_g'))   
    # print('Air gap field intensity ..........',  p.get_val('H_g'))  
    print('Equivalent air gap ...............',  p.get_val('g_eq', units='mm'))
    print('Carters Coefficient ..............',  p.get_val('carters_coef'))
    print('Ks1 ..............................',  p.get_val('stator_surface_current'))
    print('Mu_r for magnet...................',  p.get_val('Br'))

    # print('l core ..................................', p.get_val('L_core'))


