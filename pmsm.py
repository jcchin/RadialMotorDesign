# Include some modeling of iron losses
# need to add torque/speed curve and print out plot
# need to improve carters coefficient for the equivalent gap
# options for full / half wave operation
# Calculate winding factor from Lipo book, see chapter two
# output phasor diagrams?

import numpy as np
from math import pi
from openmdao.api import Problem, IndepVarComp, ExplicitComponent, ExecComp
from openmdao.api import NewtonSolver, Group, DirectSolver, NonlinearRunOnce, LinearRunOnce, view_model, BalanceComp
from Physics.sizing_comp import motor_size, motor_mass
from Physics.thermal_comp import motor_losses
from Physics.em_comp import Efficiency, torque, flux_densities, field_intsities, airgap_eq, carters_coefficient


if __name__ == "__main__":
    p = Problem()
    model = p.model

    ind = model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])

    ind.add_output('v', val=385, units='V', desc='RMS voltage')
    ind.add_output('rpm', val=5000, units='rpm', desc='Rotation speed')

    ind.add_output('k', val=0.94)                           # Input - Stacking factor assumption
    ind.add_output('k_wb', val=0.58)                        # Input - copper fill factor
    ind.add_output('gap', val=0.0010, units='m')            # Air gap distance, Need to calculate 'effective air gap, using Carters Coeff'
    ind.add_output('n_turns', val=12)                       # Input - Number of wire turns     
    ind.add_output('i', val=34.8, units='A')                # Input - RMS Current
    ind.add_output('radius_motor', val=0.0795, units='m')   # Input - Motor outer radius
    ind.add_output('t_mag', val=.0044, units='m')           # Optimize - Magnet thickness

    ind.add_output('b_ry', val=3, units='T')            # Calculate and restrain - Rotor yoke flux density
    ind.add_output('b_sy', val=3, units='T')            # Calculate and restrain - Stator yoke flux density
    ind.add_output('b_t', val=3, units='T')             # Calculate and restrain - Tooth Flux Density

    ind.add_output('n_slots', val=24)                # Input - Number of Slots
    ind.add_output('n_m', val=21)                # Input - Number of magnets

    ind.add_output('stack_length', val=0.0345, units='m')   # Optimize - Stack Length
    ind.add_output('rho', val=7845, units='kg/m**3')        # Input - Density of Hiperco-50
    ind.add_output('rho_mag', val=7500, units='kg/m**3')    # Input - Density of Magnets
    ind.add_output('l_slot_opening', val=.002, units='m')   # length of the slot opening

    ind.add_output('K_h', val=0.0073790325365744)
    ind.add_output('K_e', val=0.00000926301369333214)
    ind.add_output('f_e', val=277, units='Hz')              # Input - Frequency: 5000 RPM / 60 - sec / 3 - phases
    ind.add_output('K_h_alpha', val=1.15293258569149)
    ind.add_output('K_h_beta', val=1.72240393990502)
    ind.add_output('B_pk', val=2.4, units='T')

    ind.add_output('B_r', val= 1.39, units='T')                 # Input - Remanence magnetic flux
    ind.add_output('mu_o', val= 0.4*pi*10**-6, units='H/m')    
    ind.add_output('mu_r', val= 1.04, units='H/m')              # Input - relative magnetic permeability of ferromagnetic materials
    ind.add_output('mag_length', val= 0.005, units='m')         # Optimize - thickness of magnet
    ind.add_output('H_c', val=1353000, units='A/m')             # Input - Intrinsic Coercivity of magnets constant

    bal = BalanceComp()
    bal.add_balance('rot_or', val=0.06, units='m', use_mult=False)
    tgt = IndepVarComp(name='J', val=10, units='A/mm**2')
    model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['J'])
    
    model.add_subsystem('flux_den', flux_densities(), promotes_inputs=['B_r', 'mu_r', 't_mag', 'g_eq'], 
                                                      promotes_outputs=['B_g'])

    model.add_subsystem('fields', field_intsities(), promotes_inputs=['B_g','B_r','H_c'], 
                                                     promotes_outputs=['H_g'])

    model.add_subsystem('size', motor_size(), promotes_inputs=['n_turns','i','k_wb','radius_motor','gap','rot_or','B_g','k','b_ry','n_m','b_sy','b_t','n_slots', 't_mag'], 
                                              promotes_outputs=['w_ry', 'w_sy', 'w_t','s_d','rot_ir','sta_ir'])

    model.add_subsystem('losses', motor_losses(), promotes_inputs=['f_e', 'B_pk', 'K_h', 'K_e', 'K_h_alpha', 'K_h_beta'], 
                                                  promotes_outputs=['P_e','P_h'])
    
    model.add_subsystem(name='balance', subsys=bal)

    model.add_subsystem('mass', motor_mass(), promotes_inputs=['t_mag','rho_mag','rho','radius_motor','n_slots','sta_ir','w_t','stack_length','s_d','rot_or','rot_ir'], 
                                              promotes_outputs=['sta_mass','rot_mass','mag_mass'])

    model.add_subsystem('torque', torque(), promotes_inputs=['rot_or','B_g','i','n_m','n_turns','stack_length', 'sta_ir'], 
                                            promotes_outputs=['tq', 'stator_surface_current', 'rot_volume'])

    model.add_subsystem('mot_eff', Efficiency(), promotes_inputs=['rpm', 'i','tq','v'], 
                                                 promotes_outputs=['P_in', 'P_out'])

    model.add_subsystem('carters', carters_coefficient(), promotes_inputs=['gap', 'sta_ir', 'n_slots', 'l_slot_opening'],
                                          promotes_outputs=['mech_angle', 't_1', 'carters_coef'])

    model.add_subsystem('eq_gap', airgap_eq(), promotes_inputs=['gap', 'carters_coef', 'k_sat', 't_mag','mu_o', 'mu_r'],
                                               promotes_outputs=['g_eq'])

    model.connect('J', 'balance.rhs:rot_or')
    model.connect('balance.rot_or', 'rot_or')
    model.connect('size.J', 'balance.lhs:rot_or')

    model.linear_solver = DirectSolver()

    model.nonlinear_solver = NewtonSolver()
    model.nonlinear_solver.options['maxiter'] = 100
    model.nonlinear_solver.options['iprint'] = 0

    p.setup()
    p.final_setup()
    #p.check_partials(compact_print=True)
    # p.model.list_outputs(implicit=True)
    # p.set_solver_print(2)
    #view_model(p)
    p.run_model()

    # print('Rotor Inner Diameter..............', 2 * p.get_val('rot_ir', units='mm'))
    # print('Rotor Inner radius................',     p.get_val('rot_ir', units='mm'))

    # print('Rotor Yoke Thickness..............',  p.get_val('w_ry', units='mm'))
    # print('Magnet Thickness..................',  p.get_val('t_mag', units='mm'))

    # print('Rotor outer Diameter..............',  2 * p.get_val('rot_or', units='mm'))
    # print('Rotor outer Radius................',      p.get_val('rot_or', units='mm'))

    # print('Stator Inner Diameter.............', 2 *  p.get_val('sta_ir', units='mm'))
    # print('Stator Inner Radius...............',      p.get_val('sta_ir', units='mm'))

    # print('Slot Depth........................',  p.get_val('s_d', units='mm'))
    # print('Stator Yoke Thickness.............',  p.get_val('w_sy', units='mm'))

    # print('Motor Outer Diameter..............', 2 *  p.get_val('mass.radius_motor', units='mm'))
    # print('Motor Outer Radius................',      p.get_val('mass.radius_motor', units='mm'))

    # print('Tooth Width.......................',  p.get_val('w_t', units='mm'))

    print('Torque............................',  p['tq'])

    print('Mass of Stator....................',  p.get_val('sta_mass', units='kg'))
    print('Mass of Rotor.....................',  p.get_val('rot_mass', units='kg'))
    print('Mass of Magnets...................',  p.get_val('mag_mass', units='kg')) 
    print('Mass of Motor.....................',  p.get_val('mag_mass', units='kg') + p.get_val('rot_mass', units='kg') + p.get_val('sta_mass', units='kg'))   

    print('Current Density...................',  p.get_val('size.J'))
    print('Core Eddy Losses .................',  p.get_val('P_e'))
    print('Core Hysteresis Losses ...........',  p.get_val('P_h'))

    print('Power In  ........................',  p.get_val('P_in'))
    print('Power out  .......................',  p.get_val('P_out'))

    print('Air gap flux density .............',  p.get_val('B_g'))   
    print('Air gap field intensity ..........',  p.get_val('H_g'))  
    print('Equivalent air gap ...............',  p.get_val('g_eq', units='mm'))

    print('Carters Coefficient ..............',  p.get_val('carters_coef'))
    print('Ks1 ..............................',  p.get_val('stator_surface_current'))

