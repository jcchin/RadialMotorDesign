import unittest
from openmdao.api import Problem, Group, IndepVarComp, ExecComp, ScipyOptimizeDriver
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials

from motor_weight_optimization import motor_size, torque, motor_mass #,rotor_outer_radius,

class TestMWO(unittest.TestCase):

    def test1(self):
        p = Problem()
        model = p.model

        ind = model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])

        ind.add_output('rot_or', val=0.0615, units='m')         # Outer radius of rotor, including 5mm magnet thickness
        ind.add_output('k', val=0.97)                           # Stacking factor
        ind.add_output('n', val=16)                             # Number of wire turns      
        ind.add_output('i', val=30, units='A')                  # RMS Current
        ind.add_output('r_m', val=0.0765, units='m')

        ind.add_output('b_g', val = 1, units='T')               # Air gap flux Density    !! Flux values may represent 100% slot fill !!
        ind.add_output('b_ry', val=4, units='T')                # Rotor yoke flux density
        ind.add_output('b_sy', val=4, units='T')                # Stator yoke flux density
        ind.add_output('b_t', val=4, units='T')                 # Tooth Flux Density

        ind.add_output('n_s', val=15)                           # Number of Slots
        ind.add_output('n_m', val=16)                           # Number of poles

        ind.add_output('l_st', val=0.038, units='m')            # Stack Length
        ind.add_output('rho', val=8110.2, units='kg/m**3')      # Density of Hiperco-50

        model.add_subsystem('size', motor_size(), promotes_inputs=['rot_or','b_g','k','b_ry','n_m','b_sy','b_t','n_s'], promotes_outputs=['w_ry', 'w_sy', 'w_t','s_d','rot_ir','sta_ir'])
        model.add_subsystem('torque', torque(), promotes_inputs=['rot_or','b_g','i','n_m','n','l_st'], promotes_outputs=['tq'])
        model.add_subsystem('mass', motor_mass(), promotes_inputs=['rho','r_m','n_s','sta_ir','w_t','l_st','s_d','rot_or','rot_ir'], promotes_outputs=['sta_mass','rot_mass'])
        

        p.setup(check = False, force_alloc_complex = True)

        p.run_model()


        print('Rotor Inner Radius................',  p.get_val('rot_ir', units='mm'))
        print('Stator Inner Radius...............',  p.get_val('sta_ir', units='mm'))
        print('Motor Outer Radius................',  p.get_val('mass.r_m', units='mm'))

        print('Rotor Yoke Thickness..............',  p.get_val('w_ry', units='mm'))
        print('Slot Depth........................',  p.get_val('s_d', units='mm'))
        print('Stator Yoke Thickness.............',  p.get_val('w_sy', units='mm'))
        print('Tooth Width.......................',  p.get_val('w_t', units='mm'))

        print('Torque............................',  p['tq'])

        print('Mass of Stator....................',  p.get_val('sta_mass', units='kg'))
        print('Mass of Rotor.....................',  p.get_val('rot_mass', units='kg'))

        assert_rel_error(self, p.get_val('rot_ir', units='mm'), 53.38775857, 1e-4)
        assert_rel_error(self, p.get_val('sta_ir', units='mm'), 62.5, 1e-4)
        assert_rel_error(self, p.get_val('r_m', units='mm'), 76.5, 1e-4)
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




