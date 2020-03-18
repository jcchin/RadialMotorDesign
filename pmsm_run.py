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

import openmdao.api as om

from motor import Motor, print_motor

if __name__ == "__main__":
    p = om.Problem()
    model = p.model

    ind = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])


    # -------------------------------------------------------------------------
    #                              
    # -------------------------------------------------------------------------
    ind.add_output('radius_motor', 0.078225, units='m', desc='Motor outer radius')      # Ref motor = 0.078225
    ind.add_output('rpm', 5400, units='rpm', desc='Rotation speed')  
    ind.add_output('I', 34.35, units='A', desc='RMS Current')
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
    ind.add_output('mu_o',  0.4*pi*10**-6, units='H/m', desc='permeability of free space')   
    ind.add_output('mu_r',  1.0, units='H/m', desc='relative magnetic permeability of ferromagnetic materials')
    ind.add_output('rho', 8110.2, units='kg/m**3', desc='Density of Hiperco-50')  
    ind.add_output('rho_mag', 7500, units='kg/m**3', desc='Density of Magnets')
    # ind.add_output('rho_wire', 8940, units='kg/m**3', desc='Density of wire: Cu=8940')
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

    def connect(motor_path): 

        p.model.connect('radius_motor', f'{motor_path}.radius_motor')
        p.model.connect('rpm', f'{motor_path}.rpm')
        p.model.connect('I', f'{motor_path}.I')
        # p.model.connect('V', f'{motor_path}.V')
        p.model.connect('stack_length', f'{motor_path}.stack_length')

        p.model.connect('n_turns', f'{motor_path}.n_turns')
        p.model.connect('n_slots', f'{motor_path}.n_slots')
        p.model.connect('n_m', f'{motor_path}.n_m')

        p.model.connect('t_mag', f'{motor_path}.t_mag')
        p.model.connect('r_strand', f'{motor_path}.r_strand')
        p.model.connect('n_strands', f'{motor_path}.n_strands')
      
        p.model.connect('T_windings', f'{motor_path}.T_windings')
        p.model.connect('T_mag', f'{motor_path}.T_mag')

        # -------------------------------------------------------------------------
        #                        Material Properties and Constants
        # -------------------------------------------------------------------------
        p.model.connect('Hc_20', f'{motor_path}.Hc_20')
        p.model.connect('Br_20', f'{motor_path}.Br_20')
        p.model.connect('k_sat', f'{motor_path}.k_sat')
        p.model.connect('mu_o', f'{motor_path}.mu_o')
        p.model.connect('mu_r', f'{motor_path}.mu_r')
        p.model.connect('rho', f'{motor_path}.rho')
        p.model.connect('rho_mag', f'{motor_path}.rho_mag')
        # p.model.connect('rho_wire', f'{motor_path}.rho_wire')
        p.model.connect('resistivity_wire', f'{motor_path}.resistivity_wire')
        p.model.connect('T_coeff_cu', f'{motor_path}.T_coeff_cu')
        # p.model.connect('T_ref_wire', f'{motor_path}.T_ref_wire')
        p.model.connect('alpha_stein', f'{motor_path}.alpha_stein')
        p.model.connect('beta_stein', f'{motor_path}.beta_stein')
        p.model.connect('k_stein', f'{motor_path}.k_stein')
        p.model.connect('T_coef_rem_mag', f'{motor_path}.T_coef_rem_mag')

        # -------------------------------------------------------------------------

        p.model.connect('b_ry', f'{motor_path}.b_ry')
        p.model.connect('b_sy', f'{motor_path}.b_sy')
        p.model.connect('b_t', f'{motor_path}.b_t')
        p.model.connect('B_pk', f'{motor_path}.B_pk')
        p.model.connect('k_wb', f'{motor_path}.k_wb')

        p.model.connect('l_slot_opening', f'{motor_path}.l_slot_opening')
        p.model.connect('k', f'{motor_path}.k')
        p.model.connect('gap', f'{motor_path}.gap')

    p.model.add_subsystem('DESIGN', Motor())
    connect('DESIGN')
    # p.model.add_subsystem('OFF_DESIGN', Motor(design=False))

    p.setup()
    p.final_setup()
    #p.check_partials(compact_print=True)
    # p.model.list_outputs(implicit=True)
    # p.set_solver_print(2)
    # view_model(p)
    # quit()


    p['radius_motor'] = 0.086
    p['DESIGN.rot_or'] = 8. # initial guess
    p.run_model()
    
    print_motor(p, 'DESIGN')
    # print_motor(p, 'OFF_DESIGN')

    # print('l core ..................................', p.get_val('L_core'))

    # p.model.list_outputs(residuals=True)
    # p.check_partials(compact_print=True)
