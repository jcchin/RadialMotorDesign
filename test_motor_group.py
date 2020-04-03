import unittest
import numpy as np
from openmdao.api import Problem, Group, IndepVarComp, ExecComp, ScipyOptimizeDriver
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials

from motor import Motor
# from motor_spec_connect import motor_spec_connect

class TestMotorGroup(unittest.TestCase):
    def test_design_derivs(self):
        p = Problem()
        model = p.model
        nn = 1
    
        ind = model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])

        ind.add_output('DES:P_shaft', 14.000, units='kW', desc='shaft power out of the motor')
        ind.add_output('OD:P_shaft', 1.000*np.ones(nn), units='kW', desc='shaft power out of the motor')
    
        ind.add_output('DES:stack_length', 0.0345, units='m', desc='axial length of the motor')
        
        ind.add_output('DES:rpm', 5400, units='rpm', desc='Rotation speed')
        ind.add_output('OD:rpm', 1000*np.ones(nn), units='rpm', desc='Rotation speed')  
    
        ind.add_output('DES:I', 34.5, units='A', desc='RMS Current')
        ind.add_output('OD:I', 34.5*np.ones(nn), units='A', desc='RMS Current')
    
        ind.add_output('radius_motor', 0.078225, units='m', desc='Motor outer radius')  # Ref motor = 0.078225
    
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
        ind.add_output('mu_o',  1.2566e-6, units='H/m', desc='permeability of free space')   
        ind.add_output('mu_r',  1.0, units='H/m', desc='relative magnetic permeability of ferromagnetic materials')
        ind.add_output('rho', 8110.2, units='kg/m**3', desc='Density of Hiperco-50')  
        ind.add_output('rho_mag', 7500, units='kg/m**3', desc='Density of Magnets')
        # ind.add_output('rho_wire', 8940, units='kg/m**3', desc='Density of wire: Cu=8940')
        ind.add_output('resistivity_wire', 1.724e-8, units='ohm*m', desc='resisitivity of Cu at 20 degC') 
        ind.add_output('T_coeff_cu', 0.00393, desc='temperature coeff of copper')
        # ind.add_output('T_ref_wire', 20.0, units='C', desc='reference temperature at which winding resistivity is measured')
        ind.add_output('alpha_stein', 1.286, desc='Alpha coefficient for steinmetz, constant')
        ind.add_output('beta_stein', 1.76835, desc='Beta coefficient for steinmetz, dependent on freq')
        ind.add_output('k_stein', 0.0044, desc='k constant for steinmetz')
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

        p.model.add_subsystem('DESIGN', Motor(num_nodes=nn, design=True))

        motor_path = 'DESIGN'
        p.model.connect('radius_motor', f'{motor_path}.radius_motor')
        p.model.connect('rho', f'{motor_path}.rho')
        p.model.connect('rho_mag', f'{motor_path}.rho_mag')
        p.model.connect('b_ry', f'{motor_path}.b_ry')
        p.model.connect('b_sy', f'{motor_path}.b_sy')
        p.model.connect('b_t', f'{motor_path}.b_t')
        p.model.connect('k_wb', f'{motor_path}.k_wb')
        p.model.connect('k', f'{motor_path}.k')
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
        # p.model.connect('rho_wire', f'{motor_path}.rho_wire')
        p.model.connect('resistivity_wire', f'{motor_path}.resistivity_wire')
        p.model.connect('T_coeff_cu', f'{motor_path}.T_coeff_cu')
        # p.model.connect('T_ref_wire', f'{motor_path}.T_ref_wire')
        p.model.connect('alpha_stein', f'{motor_path}.alpha_stein')
        p.model.connect('beta_stein', f'{motor_path}.beta_stein')
        p.model.connect('k_stein', f'{motor_path}.k_stein')
        p.model.connect('T_coef_rem_mag', f'{motor_path}.T_coef_rem_mag')
        
        p.model.connect('gap', f'{motor_path}.gap')

        p.model.connect('DES:rpm', 'DESIGN.rpm')
        p.model.connect('DES:I', 'DESIGN.I')
        p.model.connect('DES:stack_length', 'DESIGN.stack_length')
        p.model.connect('DES:P_shaft', 'DESIGN.P_shaft')

        p.setup(force_alloc_complex=True)


        p['radius_motor'] = 0.086   # Set the desired radius of the motor, application specific
        p['DESIGN.rot_or'] = 6.8    # initial guess

        # p.run_model()

        # make new lines for each component

        data = p.check_partials(method='cs', compact_print=True, show_only_incorrect = True, includes='DESIGN.geometry.size')

        # can look at one component at a time with assert_check_partials
        assert_check_partials(data, atol=1e-6, rtol=1e-6)

        data = p.check_partials(method='cs', compact_print=True, show_only_incorrect = True, includes='DESIGN.em_properties.carters')
        # can look at one component at a time with assert_check_partials
        assert_check_partials(data, atol=1e-6, rtol=1e-6)

        data = p.check_partials(method='cs', compact_print=True, show_only_incorrect = True, includes='DESIGN.em_properties.equivalent_gap')
        # can look at one component at a time with assert_check_partials
        assert_check_partials(data, atol=1e-6, rtol=1e-6)

        data = p.check_partials(method='cs', compact_print=True, show_only_incorrect = True, includes='DESIGN.em_properties.gap_fields')
        # can look at one component at a time with assert_check_partials
        assert_check_partials(data, atol=1e-6, rtol=1e-6) 

        data = p.check_partials(method='cs', compact_print=True, show_only_incorrect = True, includes='DESIGN.em_properties.torque')
        # can look at one component at a time with assert_check_partials
        assert_check_partials(data, atol=1e-6, rtol=1e-6)

        data = p.check_partials(method='cs', compact_print=True, show_only_incorrect = True, includes='DESIGN.em_properties.motor_efficiency')
        # can look at one component at a time with assert_check_partials
        assert_check_partials(data, atol=1e-6, rtol=1e-6)

        # data = p.check_partials(method='cs', compact_print=True, show_only_incorrect = True, includes='DESIGN.thermal_properties.copperloss')
        # # can look at one component at a time with assert_check_partials
        # assert_check_partials(data, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
