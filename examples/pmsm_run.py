# Keep systems of equations up to date
# Magnet losses need to be completed
# Eff inputs and outputs should match up with ccblade and zappy
# need to add torque/speed curve and print out plot
# Future: Calculate winding factor from Lipo book, see chapter two
# Future: output phasor diagrams

# Make models of 20 hp (14.914 kW) motors roughly the size of a small personal quadcopter
    # Change current density based on liquid/air cooling
    # Material may matter, can do a cost / mass study
    # What voltage and current
    # What are geometry constraints: diameter, length
    # RPM from NDARC

import numpy as np
from math import pi
import matplotlib.pyplot as plt

import openmdao.api as om

from rad_motor.motor import Motor, print_motor

if __name__ == "__main__":
    p = om.Problem()
    model = p.model
    nn = 1

    ind = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])

    # -------------------------------------------------------------------------
    #                              
    # -------------------------------------------------------------------------
    ind.add_output('DES:P_shaft', 14.00, units='kW', desc='shaft power out of the motor')
    ind.add_output('OD:P_shaft', 10.000*np.ones(nn), units='kW', desc='shaft power out of the motor')

    ind.add_output('DES:stack_length', 0.0345, units='m', desc='axial length of the motor')

    ind.add_output('DES:rpm', 5000, units='rpm', desc='Rotation speed')
    ind.add_output('OD:rpm', 3000*np.ones(nn), units='rpm', desc='Rotation speed')  

    ind.add_output('radius_motor', 0.086, units='m', desc='Motor outer radius')  # Ref motor = 0.078225 --- max=0.12

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
    ind.add_output('copper_cp', 386, units='J/kg/C', desc='specific heat for copper')
    ind.add_output('hiperco_cp', 410, units='J/kg/C', desc='specific heat for hiperco')
    ind.add_output('neo_cp', 190, units='J/kg/C', desc='specific heat for neodymium')
    ind.add_output('Hc_20', -1046, units='kA/m', desc='Intrinsic Coercivity at 20 degC')     
    ind.add_output('Br_20', 1.39, units='T', desc='remnance flux density at 20 degC')           
    ind.add_output('k_sat', 1, desc='Saturation factor of the magnetic circuit due to the main (linkage) magnetic flux')
    ind.add_output('mu_o',  1.2566e-6, units='H/m', desc='permeability of free space')   
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

    # ind.add_output('l_slot_opening', .00325, units='m', desc='length of the slot opening')  # Ref motor = .00325
    ind.add_output('k', 0.94, desc='Stacking factor assumption')
    ind.add_output('gap', 0.0010, units='m', desc='Air gap distance, Need to calculate effective air gap, using Carters Coeff')

    def motor_spec_connect(motor_path): 
        if motor_path == 'DESIGN':
            p.model.connect('radius_motor', f'{motor_path}.radius_motor')
            p.model.connect('rho', f'{motor_path}.rho')
            p.model.connect('rho_mag', f'{motor_path}.rho_mag')
            p.model.connect('b_ry', f'{motor_path}.b_ry')
            p.model.connect('b_sy', f'{motor_path}.b_sy')
            p.model.connect('b_t', f'{motor_path}.b_t')
            p.model.connect('k_wb', f'{motor_path}.k_wb')
            p.model.connect('k', f'{motor_path}.k')
            p.model.connect('rho_wire', f'{motor_path}.rho_wire')
            p.model.connect('copper_cp', f'{motor_path}.copper_cp')
            p.model.connect('hiperco_cp', f'{motor_path}.hiperco_cp')
            p.model.connect('neo_cp', f'{motor_path}.neo_cp')

        p.model.connect('B_pk', f'{motor_path}.B_pk')
        p.model.connect('Br_20', f'{motor_path}.Br_20')
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
        p.model.connect('k_sat', f'{motor_path}.k_sat')
        p.model.connect('mu_o', f'{motor_path}.mu_o')
        p.model.connect('mu_r', f'{motor_path}.mu_r')
        
        p.model.connect('resistivity_wire', f'{motor_path}.resistivity_wire')
        p.model.connect('T_coeff_cu', f'{motor_path}.T_coeff_cu')
        # p.model.connect('T_ref_wire', f'{motor_path}.T_ref_wire')
        p.model.connect('alpha_stein', f'{motor_path}.alpha_stein')
        p.model.connect('beta_stein', f'{motor_path}.beta_stein')
        p.model.connect('k_stein', f'{motor_path}.k_stein')
        p.model.connect('T_coef_rem_mag', f'{motor_path}.T_coef_rem_mag')
        
        p.model.connect('gap', f'{motor_path}.gap')

    # On-Design Function, to size the motor
    p.model.add_subsystem('DESIGN', Motor(num_nodes=nn, design=True))
    motor_spec_connect('DESIGN')
    p.model.connect('DES:rpm', 'DESIGN.rpm')
    p.model.connect('DES:stack_length', 'DESIGN.stack_length')
    p.model.connect('DES:P_shaft', 'DESIGN.P_shaft')

    p.model.add_subsystem('OD1', Motor(num_nodes=nn, design=False))
    motor_spec_connect('OD1')
    p.model.connect('DESIGN.rot_or', 'OD1.rot_or')
    p.model.connect('DESIGN.w_slot', 'OD1.w_slot')
    p.model.connect('DESIGN.w_t', 'OD1.w_t')
    p.model.connect('DESIGN.sta_mass', 'OD1.sta_mass')

    p.model.connect('OD:rpm', 'OD1.rpm')
    p.model.connect('DES:stack_length', 'OD1.stack_length') # DES to OD1 to make sure it stays constant
    p.model.connect('OD:P_shaft', 'OD1.P_shaft')

    # #DOEDriver used for Off-Design Analysis by varying RPM and I
    # p.model.add_design_var('OD:rpm', lower=200, upper=5400)
    # p.model.add_design_var('OD:I', lower=10, upper=34.355)
    # p.model.add_objective('OD1.Eff')

    # steps=3   # number of levels for the full factorial generator
    # p.driver = om.DOEDriver(om.FullFactorialGenerator(levels=steps))
    # p.driver.add_recorder(om.SqliteRecorder("cases.sql"))
    # p.driver.recording_options['record_objectives'] = True
    # p.driver.recording_options['record_constraints'] = True
    # p.driver.recording_options['record_desvars'] = True
    # p.driver.recording_options['record_inputs'] = True

    p.setup()

    print('the radius is: ', p['DESIGN.geometry.size.radius_motor'])
    p['DESIGN.rot_or'] = (p['DESIGN.geometry.size.radius_motor'] *0.80 )*100   # 8.0 = (0.08*100)

    p.run_model()
    # p.run_driver()
    # p.cleanup()

    # Print Motor Stats
    print_motor(p, 'DESIGN')
    print()
    print()
    print_motor(p, 'OD1')
    # p.model.list_outputs(residuals=True)
    # p.check_partials(compact_print=True)

    # Case Recorder for the Motor off design case
#     cr = om.CaseReader("cases.sql")  
#     system_cases = cr.list_cases('driver')

#     eff_val=[]
#     rpm_val=[]
#     I_val=[]
#     I_array=[]

#     # Save off arrays from the .sql file for the efficiency mapping
#     for i in range(0, len(system_cases)):
#         case_array = cr.get_case(i)
#         eff_val.append(case_array.outputs['OD1.Eff'][0])
#         rpm_val.append(case_array['OD:rpm'][0])
#         I_val.append(case_array['OD:I'][0])

# # Printing Eff Plot: Reshape arrays to a useable format for meshgrid
# eff_val = np.array([eff_val[i:i + steps] for i in range(0, len(eff_val), steps)])
# rpm_val = np.array(rpm_val[:steps])
# I_val = I_val[::steps]

# # Begin plotting the efficiency map
# X, Y = np.meshgrid(rpm_val, I_val)
# z = eff_val
# a_ratio = 5400/34.355

# # levels = np.array([.90, .93, .95, .955, .96, .965, .968, .97])
# contours = plt.contour(X, Y, z, colors='Black') #levels,
# plt.clabel(contours, inline=True, fontsize=10)
# # plt.imshow(z, aspect=a_ratio, extent=[200, 5400, 10, 34.35],  origin='lower', cmap='Reds') 
# # plt.colorbar()

# plt.xlabel('RPM')
# plt.ylabel('Current')
# plt.title('Efficiency Plot')
# plt.savefig("Efficiency_Plot.pdf")

