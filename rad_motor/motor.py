import openmdao.api as om

from rad_motor.electromagnetics.em_group import EmGroup
from rad_motor.thermal.thermal_group import ThermalGroup
from rad_motor.sizing.size_group import SizeGroup

class Motor(om.Group): 


    def initialize(self): 
        self.options.declare('design', default=True, types=bool)
        self.options.declare('num_nodes', types=int)


    def setup(self): 
        nn = self.options['num_nodes']

        self.add_subsystem('thermal_properties', ThermalGroup(num_nodes=nn), promotes_inputs=['B_pk', 'alpha_stein', 'beta_stein', 'k_stein', 'rpm', 'sta_mass', 
                                                                                              'resistivity_wire', 'stack_length', 'n_slots', 'n_strands', 'I_required', 
                                                                                              'n_m', 'mu_o', 'f_e', 'n_turns', 'T_coeff_cu', 'T_windings', 'r_strand', 'mu_r'],
                                                                            promotes_outputs=['A_cu', 'r_litz', 'P_steinmetz', 'steinmetz', 'P_dc', 'P_ac', 'P_wire', 'L_wire', 'R_dc',
                                                                                              'skin_depth', 'temp_resistivity', 'f_e', 'Q_total'])


        self.add_subsystem('em_properties', EmGroup(num_nodes=nn), promotes_inputs=['w_slot', 'w_t', 'T_coef_rem_mag', 'T_mag',            
                                                                                    'gap', 'carters_coef', 'k_sat', 'stack_length',                  
                                                                                    'Br', 'Br_20', 'mu_r', 'g_eq', 't_mag',          
                                                                                    'B_g', 'n_m', 'n_turns', 'rot_or',  'rpm',  
                                                                                    'P_wire', 'P_steinmetz', 'P_shaft', 'Tq_shaft', 'omega'],
                                                                  promotes_outputs=['Br', 'carters_coef', 'Tq_shaft', 'I_required',             
                                                                                    'g_eq','omega', 'P_in', 'Eff', 'B_g'])                                                     
  
        if self.options['design']: 

            self.add_subsystem('geometry', SizeGroup(num_nodes=nn), promotes_inputs=['gap', 'B_g', 'k', 'b_ry', 'n_m', 'b_sy', 'b_t', 'n_turns', 'I_required', 'k_wb',
                                                                     'rho', 'radius_motor', 'n_slots', 'sta_ir', 'w_t', 'stack_length',
                                                                     's_d', 'rot_or', 'rot_ir', 't_mag', 'rho_mag', 'L_wire', 'rho_wire', 'r_litz',
                                                                     'mag_mass', 'sta_mass', 'rot_mass', 'wire_mass', 'motor_mass', 'hiperco_cp', 'copper_cp', 'neo_cp'],
                                                   promotes_outputs=['J', 'w_ry', 'w_sy', 'w_t', 'sta_ir', 'rot_ir', 's_d', 
                                                                     'mag_mass', 'sta_mass', 'rot_mass', 'wire_mass', 'slot_area', 'w_slot', 'motor_mass', 'cp_motor'])

            bal = om.BalanceComp(num_nodes=nn)
            bal.add_balance('rot_or', val=0.068, units='m', eq_units='A/mm**2', lower=1e-4)
            tgt = om.IndepVarComp(name='J_tgt', val=17.51, units='A/mm**2')

            self.add_subsystem(name='target', subsys=tgt, promotes_outputs=['J_tgt'])
            self.add_subsystem(name='balance', subsys=bal, promotes_outputs=['rot_or'])

            self.connect('J_tgt', 'balance.rhs:rot_or')
            self.connect('J', 'balance.lhs:rot_or')

        self.linear_solver = om.DirectSolver()

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['maxiter'] = 100
        newton.options['iprint'] = 2
        newton.options['solve_subsystems'] = True

        # ls = newton.linesearch = om.ArmijoGoldsteinLS()
        # ls.options['maxiter'] = 3
        # ls.options['iprint'] = 2
        # ls.options['print_bound_enforce'] = True

        ls = newton.linesearch = om.BoundsEnforceLS()
        ls.options['print_bound_enforce'] = True


def print_motor(prob, motor_path=''): 

    print('***'*30)
    print(f'* Data for motor: {motor_path}')
    print('***'*30)

    if motor_path: 
        motor_path += "."

    # p.model.run_apply_nonlinear()
    if motor_path == 'DESIGN.':


        print('Motor Outer Radius................',      prob.get_val('DESIGN.geometry.mass.radius_motor', units='mm'))
        # print('Stator Inner Radius...............',      prob.get_val('DESIGN.sta_ir', units='mm'))
        print('Rotor Outer Radius................',     prob.get_val('DESIGN.rot_or', units='mm'))
        # print('Rotor Inner Radius................',     prob.get_val('DESIGN.rot_ir', units='mm'))
        print('Mass of Stator....................',  prob.get_val('DESIGN.sta_mass', units='kg'))
        # print('Mass of Rotor....................',  prob.get_val('DESIGN.rot_mass', units='kg'))
        # print('Mass of Magnets....................',  prob.get_val('DESIGN.mag_mass', units='kg'))
        # print('Mass of Copper....................',  prob.get_val('DESIGN.wire_mass', units='kg'))
        print('Mass of Motor.....................',  prob.get_val('DESIGN.motor_mass'))
        # print('Length of Phase...................',  prob.get_val('DESIGN.L_wire', units='m'))
        # print('Air gap flux density..............',  prob.get_val('DESIGN.B_g', units='T'))
        # print('\n')
        # print('#---------------MOTOR CAD INPUTS -----------------#')
        # print('Stator Lam Diameter...............',  prob.get_val('DESIGN.geometry.mass.radius_motor', units='mm')*2)
        # print('Stator Bore.......................',  prob.get_val('DESIGN.sta_ir', units='mm')*2)
        print('Tooth Width.......................',  prob.get_val('DESIGN.w_t', units='mm'))
        print('Width of the slot.................',  prob.get_val('DESIGN.w_slot'))
        # print('Slot Depth........................',  prob.get_val('DESIGN.s_d', units='mm'))
        # print('Shaft Diameter....................',  prob.get_val('DESIGN.rot_ir', units='mm')*2)
        # print('Copper temperature resistivity....',  prob.get_val('DESIGN.temp_resistivity'))
        print('Specific Heat.....................',  prob.get_val('DESIGN.cp_motor'))
        # print('#-------------------------------------------------#')
        # print('\n')
    # print('Iron losses.............',   prob.get_val(f'{motor_path}P_steinmetz') * prob.get_val(f'{motor_path}sta_mass'))
    # print('DC Winding  Losses......',   prob.get_val(f'{motor_path}P_dc'))
    # print('AC Winding  Losses......',   prob.get_val(f'{motor_path}P_ac'))
    # print('DC Resistance...........',   prob.get_val(f'{motor_path}R_dc', units='ohm'))
    print('Total Losses Add/Sub....',   prob.get_val(f'{motor_path}Q_total'))
    # print('Steinmetz value.........',   prob.get_val(f'{motor_path}steinmetz'))
    # print('Electrical Frequency....',  prob.get_val(f'{motor_path}f_e'))
    print('Overall Efficiency......',   prob.get_val(f'{motor_path}Eff'))

    print('Power Required...........',  prob.get_val(f'{motor_path}P_in'))
    
    print('Torque............................',  prob.get_val(f'{motor_path}Tq_shaft'))
    print('Required Current..................',  prob.get_val(f'{motor_path}I_required'))

    # print('--------------FIELDS--------------')
    # print('Air gap flux density .............',  prob.get_val(f'{motor_path}B_g'))   
    # print('Equivalent air gap ...............',  prob.get_val(f'{motor_path}g_eq', units='mm'))
    # print('Carters Coefficient ..............',  prob.get_val(f'{motor_path}carters_coef'))
    # print('Mu_r for magnet...................',  prob.get_val(f'{motor_path}Br'))
