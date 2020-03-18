import openmdao.api as om

from electromagnetics.em_group import EmGroup
from thermal.thermal_group import ThermalGroup
from sizing.size_group import SizeGroup

class Motor(om.Group): 


    def initialize(self): 
        self.options.declare('design', default=True, types=bool)


    def setup(self): 

        self.add_subsystem('thermal_properties', ThermalGroup(), promotes_inputs=[ 'B_pk', 'alpha_stein', 'beta_stein', 'k_stein', 'rpm', 
                                                                                   'resistivity_wire', 'stack_length', 'n_slots', 'n_strands', 
                                                                                   'n_m', 'mu_o', 'n_turns', 'T_coeff_cu', 'I', 'T_windings', 'r_strand', 'mu_r'],
                                                                                   # 'K_h_alpha', 'K_h_beta', 'K_h', 'K_e', D_b', 'F_b', 'alpha', 'gap', 'k', 'mu_a', 'muf_b', 'n_m', 'rho_a', 'rot_ir', 'rot_or', 'rpm', 'stack_length'],
                                                                  promotes_outputs=[
                                                                                    # 'L_core','L_emag', 'L_ewir', 'L_airg', 'L_airf', 'L_bear','L_total',
                                                                                    'A_cu', 'r_litz', 'P_steinmetz', 'P_dc', 'P_ac', 'P_wire', 'L_wire', 'R_dc', 'skin_depth', 'temp_resistivity', 'f_e'])


        self.add_subsystem('em_properties', EmGroup(), promotes_inputs=['T_coef_rem_mag', 'T_mag', 'I', 'rpm', 'mu_r', 'g_eq', 't_mag', 'Br_20',
                                                                         'gap', 'sta_ir', 'n_slots', 'l_slot_opening', 't_mag',
                                                                         'carters_coef', 'k_sat', 'mu_o', 'P_wire', 'P_steinmetz',
                                                                         'B_g', 'n_m', 'n_turns', 'stack_length', 'rot_or', 's_d', 'w_t', 'slot_area', 't_mag', 'w_slot'], 
                                                        promotes_outputs=['P_in', 'P_out', 'B_g', 
                                                                          'mech_angle', 't_1', 'carters_coef',
                                                                          'g_eq', 'g_eq_q', 'Br', 'Eff', 
                                                                          'Tq', 'rot_volume', 'stator_surface_current'])
        
       
        self.add_subsystem('geometry', SizeGroup(), promotes_inputs=['gap', 'B_g', 'k', 'b_ry', 'n_m',
                                                                    'b_sy', 'b_t', 'n_turns', 'I', 'k_wb',
                                                                    'rho', 'radius_motor', 'n_slots', 'sta_ir', 'w_t', 'stack_length',
                                                                    's_d', 'rot_or', 'rot_ir', 't_mag', 'rho_mag'],
                                                     promotes_outputs=['J', 'w_ry', 'w_sy', 'w_t', 'sta_ir', 'rot_ir', 's_d',
                                                                     'mag_mass', 'sta_mass', 'rot_mass', 'slot_area', 'w_slot'])


        if self.options['design']: 

            bal = om.BalanceComp()
            bal.add_balance('rot_or', val=0.05, units='cm', eq_units='A/mm**2', lower=1e-4)#, use_mult=True, mult_val=0.5)
            tgt = om.IndepVarComp(name='J_tgt', val=10.47, units='A/mm**2')

            self.add_subsystem(name='target', subsys=tgt, promotes_outputs=['J_tgt'])
            self.add_subsystem(name='balance', subsys=bal, promotes_outputs=['rot_or'])

            self.connect('J_tgt', 'balance.rhs:rot_or')
            self.connect('J', 'balance.lhs:rot_or')

    
        self.linear_solver = om.DirectSolver()

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['maxiter'] = 15
        newton.options['iprint'] = 2
        newton.options['solve_subsystems'] = True

        # ls = newton.linesearch = om.ArmijoGoldsteinLS()
        # ls.options['maxiter'] = 3
        # ls.options['iprint'] = 2
        # ls.options['print_bound_enforce'] = True

        ls = newton.linesearch = om.BoundsEnforceLS()
        ls.options['print_bound_enforce'] = True
