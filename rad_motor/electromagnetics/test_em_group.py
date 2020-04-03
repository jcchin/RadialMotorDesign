from __future__ import print_function, division, absolute_import

import unittest
import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, DirectSolver, NewtonSolver
from openmdao.utils.assert_utils import assert_check_partials, assert_rel_error

from em_group import EmGroup

class TestEmGroup(unittest.TestCase):

    def test_values(self):
        n=1

        p = Problem(model=Group())

        # Only need to add output to variables that impact the test case
        # Need to test other component also
        ivc = IndepVarComp()
        ivc.add_output('gap', val=.001, units='m')
        ivc.add_output('w_slot', val=.01508251, units='m')
        ivc.add_output('w_t', val=.0047909798, units='m')
        ivc.add_output('t_mag', val=.0044, units='m')
        ivc.add_output('Br_20', val=1.39, units='T')
        ivc.add_output('T_coef_rem_mag', val=.00393)
        ivc.add_output('T_mag', val=100, units='C')
        ivc.add_output('k_sat', val=1)
        ivc.add_output('mu_r', val=1, units='H/m')
        ivc.add_output('n_m', val=20)
        ivc.add_output('n_turns', val=12)
        ivc.add_output('I', val=34.35, units='A')
        ivc.add_output('rot_or', val=.06841554332, units='m')
        ivc.add_output('P_shaft', val=14.0, units='kW')
        ivc.add_output('rpm', val=2000, units='rpm')
        ivc.add_output('stack_length', val=.0345, units='m')
        ivc.add_output('P_wire', val=396, units='W')
        ivc.add_output('P_steinmetz', val=260, units='W')

        p.model.add_subsystem(name='ivc', subsys=ivc, promotes_outputs=['*'])
        p.model.add_subsystem(name='em', subsys=EmGroup(num_nodes=n))

        p.model.connect('P_shaft', 'em.P_shaft')
        p.model.connect('P_wire', 'em.P_wire')
        p.model.connect('P_steinmetz', 'em.P_steinmetz')

        # p.model.linear_solver = DirectSolver()

        p.setup()

        p['P_shaft'] = 10 # units= 'kW'

        p.run_model()

        print('Power in to motor', p['em.P_in'])
        print('Torque shaft as fun(rpm) ', p['em.Tq_shaft'])
        print('Omega ', p['em.omega'])

        # better method in openmdao
        np.testing.assert_almost_equal(p['em.P_in'], 
                                       p['em.Tq_shaft']*p['em.omega']+p['P_wire']+p['P_steinmetz'], 
                                       decimal=6)

    def test_derivs(self):
        n=1
        p=Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output('gap', val=.001, units='m')
        ivc.add_output('w_slot', val=.01508251, units='m')
        ivc.add_output('w_t', val=.0047909798, units='m')
        ivc.add_output('t_mag', val=.0044, units='m')
        ivc.add_output('Br_20', val=1.39, units='T')
        ivc.add_output('T_coef_rem_mag', val=.00393)
        ivc.add_output('T_mag', val=100, units='C')
        ivc.add_output('k_sat', val=1)
        ivc.add_output('mu_r', val=1, units='H/m')
        ivc.add_output('n_m', val=20)
        ivc.add_output('n_turns', val=12)
        ivc.add_output('I', val=34.35, units='A')
        ivc.add_output('rot_or', val=.06841554332, units='m')
        ivc.add_output('P_shaft', val=14.0, units='kW')
        ivc.add_output('rpm', val=2000, units='rpm')
        ivc.add_output('stack_length', val=.0345, units='m')
        ivc.add_output('P_wire', val=396, units='W')
        ivc.add_output('P_steinmetz', val=260, units='W')

        p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])
        p.model.add_subsystem(name='em', subsys=EmGroup(num_nodes=n))

        p.model.connect('P_shaft', 'em.P_shaft')
        p.model.connect('P_wire', 'em.P_wire')
        p.model.connect('P_steinmetz', 'em.P_steinmetz')

        p.model.linear_solver = DirectSolver()
        newton = p.model.nonlinear_solver = NewtonSolver()
        newton.options['solve_subsystems'] = True

        p.setup(force_alloc_complex=True)

        p['P_shaft'] = 10

        p.run_model()

        data = p.check_partials(method='cs', compact_print=True, show_only_incorrect = True)
        assert_check_partials(data, atol=1e-6, rtol=1e-6)

if __name__ == '__main__':
    unittest.main()


