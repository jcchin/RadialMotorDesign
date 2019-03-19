import unittest
from openmdao.api import Problem, Group, IndepVarComp, ExecComp, ScipyOptimizeDriver
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials

from motor_weight_optimization import rotor_outer_radius, rotor_yoke_width, stator_yoke_width, tooth_width, torque, stator_mass, rotor_mass

class TestMWO(unittest.TestCase):

    def test1(self):
        p = Problem()
        model = p.model

        ind = model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])

        ind.add_output('rot_or', val=0.0615, units='m')         # Outer radius of rotor, including 5mm magnet thickness
        ind.add_output('k', val=0.97)                           # Stacking factor
        ind.add_output('n', val=16)                             # Number of wire turns      
        ind.add_output('i', val=30, units='A')                  # RMS Current

        ind.add_output('b_g', val = 1, units='T')               # Air gap flux Density    !! Flux values may represent 100% slot fill !!
        ind.add_output('b_ry', val=4, units='T')                # Rotor yoke flux density
        ind.add_output('b_sy', val=4, units='T')                # Stator yoke flux density
        ind.add_output('b_t', val=4, units='T')             # Tooth Flux Density

        ind.add_output('n_s', val=15)                           # Number of Slots
        ind.add_output('n_m', val=16)                           # Number of poles

        ind.add_output('l_st', val=0.038, units='m')            # Stack Length
        ind.add_output('rho', val=8110.2, units='kg/m**3')      # Density of Hiperco-50


        model.add_subsystem('rotor_yw', rotor_yoke_width(), promotes_inputs=['rot_or','b_g','k','b_ry','n_m'], promotes_outputs=['w_ry'])
        model.add_subsystem('stator_yoke_width', stator_yoke_width(), promotes_inputs=['rot_or','b_g','k','b_sy','n_m'], promotes_outputs=['w_sy'])
        model.add_subsystem('slot_depth', ExecComp('s_d = .0765-rot_or-0.001 - w_sy',w_sy={'units':'m'}, s_d={'units':'m'},rot_or={'units':'m'}), promotes_inputs=['rot_or','w_sy'], promotes_outputs=['s_d'])
        model.add_subsystem('motor_radius', ExecComp('r_m = rot_or + .001 + s_d + w_sy',w_sy={'units':'m'},r_m={'units':'m'}, rot_or={'units':'m'}, s_d={'units':'m'}), promotes_inputs=['rot_or','s_d','w_sy'], promotes_outputs=['r_m'])
        model.add_subsystem('rotor_in_radius', ExecComp('rot_ir = rot_or - w_ry - .005',rot_or={'units':'m'}, rot_ir={'units':'m'}, w_ry={'units':'m'}), promotes_inputs=['rot_or', 'w_ry'], promotes_outputs=['rot_ir'])
        model.add_subsystem('stator_in_radius', ExecComp('sta_ir = rot_or + .001', rot_or={'units':'m'},sta_ir={'units':'m'}), promotes_inputs=['rot_or'], promotes_outputs=['sta_ir'])
        model.add_subsystem('tooth_width', tooth_width(), promotes_inputs=['rot_or','b_t','k','n_s','b_g'], promotes_outputs=['w_t'])
        model.add_subsystem('torque', torque(), promotes_inputs=['rot_or','b_g', 'i','n_m','n','l_st'], promotes_outputs=['tq'])
        model.add_subsystem('stator_mass', stator_mass(), promotes_inputs=['rho','r_m','n_s','sta_ir','w_t','l_st','s_d'], promotes_outputs=['sta_mass'])
        model.add_subsystem('rotor_mass', rotor_mass(), promotes_inputs=['rho','rot_or','rot_ir','l_st'], promotes_outputs=['rot_mass'])

        # model.add_subsystem('motor_radius_prime', ExecComp('r_m_p = rot_or + .005 + .001 + s_d + w_sy',r_m_p={'units':'m'}, rot_or={'units':'m'}, s_d={'units':'m'}, w_sy={'units':'m'}), promotes_inputs=['rot_or','s_d','w_sy'], promotes_outputs=['r_m_p'])
        # model.add_subsystem('mass_stator', mass_stator(), promotes_inputs=['rho','r_m','n_s','sta_ir','w_t','l_st'], promotes_outputs=['weight']
        # model.add_subsystem('stmass', ExecComp('mass = l_st * ((math.pi * r_m**2)-(math.pi * sta_ir**2)+(n_s*(w_t*1.2)))', l_st={'units':'m'},r_m={'units':'m'},sta_ir={'units':'m'},w_t={'units':'m'}), promotes_inputs=['l_st','r_m','sta_ir','n_s','w_t'], promotes_outputs=['mass']


        p.setup(check = False, force_alloc_complex = True)

        p.run_model()


        print('Rotor Inner Radius................',  p.get_val('rot_ir', units='mm'))
        print('Stator Inner Radius...............',  p.get_val('sta_ir', units='mm'))
        print('Motor Outer Radius................',  p.get_val('motor_radius.r_m', units='mm'))

        print('Rotor Yoke Thickness..............',  p.get_val('w_ry', units='mm'))
        print('Slot Depth........................',  p.get_val('s_d', units='mm'))
        print('Stator Yoke Thickness.............',  p.get_val('w_sy', units='mm'))
        print('Tooth Width.......................',  p.get_val('w_t', units='mm'))

        print('Torque............................',  p['tq'])

        print('Mass of Stator....................',  p.get_val('sta_mass', units='kg'))
        print('Mass of Rotor.....................',  p.get_val('rot_mass', units='kg'))

        assert_rel_error(self, p.get_val('rot_ir', units='mm'), 53.38775857, 1e-4)
        assert_rel_error(self, p.get_val('sta_ir', units='mm'), 62.5, 1e-4)
        assert_rel_error(self, p.get_val('motor_radius.r_m', units='mm'), 76.5, 1e-4)
        assert_rel_error(self, p.get_val('w_ry', units='mm'), 3.11224143, 1e-4)
        assert_rel_error(self, p.get_val('s_d', units='mm'), 10.88775857, 1e-4)
        assert_rel_error(self, p.get_val('w_sy', units='mm'), 3.11224143, 1e-4)
        assert_rel_error(self, p.get_val('w_t', units='mm'), 6.63944839, 1e-4)
        assert_rel_error(self, p.get_val('tq', units='N*m'), 24.4094976, 1e-4)
        assert_rel_error(self, p.get_val('sta_mass', units='kg'), 0.95291846, 1e-4)
        assert_rel_error(self, p.get_val('rot_mass', units='kg'), 0.90235963, 1e-4)


        #cpd = p.check_partials(compact_print = True, show_only_incorrect = True, method = "cs")
        #assert_check_partials(cpd, atol = 1e-6, rtol = 1e-6)

if __name__ == "__main__":

    unittest.main()




