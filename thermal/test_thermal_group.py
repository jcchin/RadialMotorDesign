from __future__ import print_function, division, absolute_import

import unittest
import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, DirectSolver, NewtonSolver
from openmdao.utils.assert_utils import assert_check_partials, assert_rel_error

# from em_group import EmGroup

class TestThermalGroup(unittest.TestCase):

    def test_values(self):
        n=1

        p = Problem(model=Group())

        # Only need to add output to variables that impact the test case
        # Need to test other component also
        ivc = IndepVarComp()
        ivc.add_output('rpm', 4000*np.ones(n), units='rpm', desc='Rotation speed')
        ivc.add_output('I_peak', 50*np.ones(n), units='A', desc='Peak current')
        ivc.add_output('resistivity_wire', 1.724e-8, units='ohm*m', desc='resisitivity of Cu at 20 degC' )
        ivc.add_output('stack_length', 0.0345, units='m', desc='axial length of the motor')
        ivc.add_output('n_slots', 24, desc='Number of Slots')
        ivc.add_output('n_turns', 12, desc='Number of wire turns')
        ivc.add_output('T_coeff_cu', 0.00393, desc='temperature coeff of copper')
        ivc.add_output('I', 34.5, units='A', desc='RMS Current')
        ivc.add_output('T_windings', 150, units='C', desc='operating temperature of windings')
        ivc.add_output('r_strand', 0.0001605, units='m', desc='28 AWG radius of one strand of litz wire')
        ivc.add_output('n_m', 20, desc='Number of magnets')
        ivc.add_output('mu_o',  1.2566e-6, units='H/m', desc='permeability of free space')
        ivc.add_output('mu_r',  1.0, units='H/m', desc='relative magnetic permeability of ferromagnetic materials')
        ivc.add_output('n_strands', 41, desc='number of strands in hand for litz wire')
        ivc.add_output('alpha_stein', 1.286, desc='Alpha coefficient for steinmetz, constant')
        ivc.add_output('B_pk', 2.4, units='T', desc='Peak flux density for Hiperco-50')
        ivc.add_output('beta_stein', 1.76835, desc='Beta coefficient for steinmentz, dependent on freq')
        ivc.add_output('k_stein', 0.0044, desc='k constant for steinmentz')
        ivc.add_output('motor_mass')