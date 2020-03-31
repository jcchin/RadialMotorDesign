import openmdao.api as om

from electromagnetics.em_group import EmGroup
from thermal.thermal_group import ThermalGroup
from sizing.size_group import SizeGroup

class Motor(om.Group): 


    def initialize(self): 
        self.options.declare('design', default=True, types=bool)
        self.options.declare('num_nodes', types=int)


    def setup(self): 
        nn = self.options['num_nodes']

        self.add_subsystem('thermal_properties', ThermalGroup(num_nodes=nn), promotes_inputs=[ 'B_pk', 'alpha_stein', 'beta_stein', 'k_stein', 'rpm', 
                                                                                   'resistivity_wire', 'stack_length', 'n_slots', 'n_strands','motor_mass', 
                                                                                   'n_m', 'mu_o', 'f_e', 'n_turns', 'T_coeff_cu', 'I', 'T_windings', 'r_strand', 'mu_r'],
                                                                  promotes_outputs=['A_cu', 'r_litz', 'P_steinmetz', 'P_dc', 'P_ac', 'P_wire', 'L_wire', 'R_dc',
                                                                                    'skin_depth', 'temp_resistivity', 'f_e'])


        self.add_subsystem('em_properties', EmGroup(num_nodes=nn), promotes_inputs=['w_slot',             
                                                                        'w_t', 'T_coef_rem_mag', 'T_mag',            
                                                                        'gap', 'carters_coef', 'k_sat', 'stack_length',                  
                                                                        'Br', 'mu_r', 'g_eq', 't_mag',          
                                                                        'B_g', 'n_m', 'n_turns', 'I', 'rot_or',  'rpm',  
                                                                        'P_wire', 'P_steinmetz', 'P_shaft', 'Tq_shaft', 'omega'],       #  'l_slot_opening',  
                                                        promotes_outputs=['Br', 'carters_coef', 'Tq_shaft', 'Tq_max',             
                                                                          'g_eq','omega', 'P_in', 'Eff',                               
                                                                          'B_g'])        # 'mech_angle', 't_1',                                                   
                                                                                                                                     
        
        self.add_subsystem('geometry', SizeGroup(), promotes_inputs=['gap', 'B_g', 'k', 'b_ry', 'n_m',
                                                                    'b_sy', 'b_t', 'n_turns', 'I', 'k_wb',
                                                                    'rho', 'radius_motor', 'n_slots', 'sta_ir', 'w_t', 'stack_length',
                                                                    's_d', 'rot_or', 'rot_ir', 't_mag', 'rho_mag'],
                                                     promotes_outputs=['J', 'w_ry', 'w_sy', 'w_t', 'sta_ir', 'rot_ir', 's_d', 'motor_mass',
                                                                     'mag_mass', 'sta_mass', 'rot_mass', 'slot_area', 'w_slot'])


        if self.options['design']: 

            bal = om.BalanceComp(num_nodes=nn)
            bal.add_balance('rot_or', val=0.05, units='cm', eq_units='A/mm**2', lower=1e-4)#, use_mult=True, mult_val=0.5)
            tgt = om.IndepVarComp(name='J_tgt', val=10.47, units='A/mm**2')

            self.add_subsystem(name='target', subsys=tgt, promotes_outputs=['J_tgt'])
            self.add_subsystem(name='balance', subsys=bal, promotes_outputs=['rot_or'])

            self.connect('J_tgt', 'balance.rhs:rot_or')
            self.connect('J', 'balance.lhs:rot_or')

    
        self.linear_solver = om.DirectSolver()

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['maxiter'] = 25
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
    print('-----------GEOMETRY---------------')
    # print('Rotor Inner Diameter..............', 2 * p.get_val('rot_ir', units='mm'))
    print('Rotor Inner Radius................',     prob.get_val(f'{motor_path}rot_ir', units='mm'))
    # print('Rotor Yoke Thickness..............',  prob.get_val('w_ry', units='mm'))
    # print('Magnet Thickness..................',  prob.get_val('t_mag', units='mm'))

    # print('Rotor outer Diameter..............',  2 * prob.get_val('rot_or', units='mm'))
    # print('Rotor Outer Radius................',      prob.get_val(f'{motor_path}rot_or', units='mm'))
    print('Rotor Outer Radius................',     prob.get_val(f'{motor_path}em_properties.torque.rot_or', units='mm'))

    # print('Stator Inner Diameter.............', 2 *  prob.get_val('sta_ir', units='mm'))
    print('Stator Inner Radius...............',      prob.get_val(f'{motor_path}sta_ir', units='mm'))

    print('Slot Depth........................',  prob.get_val(f'{motor_path}s_d', units='mm'))
    print('Stator Yoke Thickness.............',  prob.get_val(f'{motor_path}w_sy', units='mm'))

    # print('Motor Outer Diameter..............', 2 *  prob.get_val(f'{motor_path}mass.radius_motor', units='mm'))
    print('Motor Outer Radius................',      prob.get_val(f'{motor_path}geometry.mass.radius_motor', units='mm'))

    print('Tooth Width.......................',  prob.get_val(f'{motor_path}w_t', units='mm'))
    print('Radius of litz wire ..............', prob.get_val(f'{motor_path}r_litz', units='m'))
    print('Length of Windings................',  prob.get_val(f'{motor_path}L_wire', units='m'))
    print('Slot Area.........................', prob.get_val(f'{motor_path}slot_area', units='m**2'))
    print('Slot Width........................', prob.get_val(f'{motor_path}w_slot'))
    print('Copper area in one slot...........', prob.get_val(f'{motor_path}A_cu', units='m**2'))
    print('Copper Slot Fill Percentage.......', prob.get_val(f'{motor_path}A_cu') / prob.get_val(f'{motor_path}slot_area'))

    print('--------------MASS----------------')
    print('Mass of Stator....................',  prob.get_val(f'{motor_path}sta_mass', units='kg'))
    print('Mass of Rotor.....................',  prob.get_val(f'{motor_path}rot_mass', units='kg'))
    print('Mass of Magnets...................',  prob.get_val(f'{motor_path}mag_mass', units='kg')) 
    print('Mass of Motor.....................',  prob.get_val(f'{motor_path}motor_mass', units='kg') + prob.get_val(f'{motor_path}rot_mass', units='kg') + prob.get_val(f'{motor_path}sta_mass', units='kg'))   
    

    print('--------------LOSSES-------------')
    print('Current Density.........',   prob.get_val(f'{motor_path}J'))
    print('Iron losses.............',   prob.get_val(f'{motor_path}P_steinmetz') * prob.get_val(f'{motor_path}sta_mass'))
    print('DC Winding  Losses......',   prob.get_val(f'{motor_path}P_dc'))
    print('AC Winding  Losses......',   prob.get_val(f'{motor_path}P_ac'))
    print('TOTAL Winding  Losses...',   prob.get_val(f'{motor_path}P_wire'))
    print('Total Losses............',   prob.get_val(f'{motor_path}P_steinmetz') * prob.get_val(f'{motor_path}sta_mass') + prob.get_val(f'{motor_path}P_wire'))
    print('Overall Efficiency......',   prob.get_val(f'{motor_path}Eff'))
    # print('Skin Depth..............',   prob.get_val(f'{motor_path}skin_depth', units='mm'))
    # print('Temp Dependent Resistivity.......', prob.get_val(f'{motor_path}temp_resistivity', units='ohm*m'))

    print('--------------EM PERF-------------')
    print('Power In  ........................',  prob.get_val(f'{motor_path}P_in'))
    # print('Power out  .......................',  prob.get_val(f'{motor_path}P_shaft'))
    print('Electrical Frequency..............',  prob.get_val(f'{motor_path}f_e'))
    print('Torque............................',  prob.get_val(f'{motor_path}Tq_shaft'))
    print('Max Torque........................',  prob.get_val(f'{motor_path}Tq_max'))

    print('--------------FIELDS--------------')
    print('Air gap flux density .............',  prob.get_val(f'{motor_path}B_g'))   
    # print('Air gap field intensity ..........',  prob.get_val(f'{motor_path}H_g'))  
    print('Equivalent air gap ...............',  prob.get_val(f'{motor_path}g_eq', units='mm'))
    print('Carters Coefficient ..............',  prob.get_val(f'{motor_path}carters_coef'))
    print('Mu_r for magnet...................',  prob.get_val(f'{motor_path}Br'))
