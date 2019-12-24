from __future__ import absolute_import

import numpy as np
from math import pi
from openmdao.api import Problem, IndepVarComp, ExplicitComponent, ExecComp
from openmdao.api import NewtonSolver, Group, DirectSolver, NonlinearRunOnce, LinearRunOnce, view_model, BalanceComp, ScipyOptimizeDriver

# from ..size_comp import MotorSizeComp
from electromagnetics.efficiency.overall_eff import EfficiencyComp
from electromagnetics.fields.airgap_fields import GapFieldsComp, CartersComp, GapEquivalentComp
from electromagnetics.torque.torque_comp import TorqueComp



class EmGroup(Group):
    def setup(self):
        ivc = IndepVarComp()

        self.add_subsystem(name='motor_efficiency', 
                           subsys=EfficiencyComp(),
                           promotes_inputs=['I', 'Tq', 'V', 'rpm'],
                           promotes_outputs=['P_in', 'P_out'])

        self.add_subsystem(name='gap_fields',
                           subsys=GapFieldsComp(),
                           promotes_inputs=['mu_r', 'g_eq', 't_mag', 'Hc_20', 'Br_20'],       
                           promotes_outputs=['B_g', 'H_g'])

        self.add_subsystem(name='carters',
                           subsys=CartersComp(),
                           promotes_inputs=['gap', 'sta_ir', 'n_slots', 'l_slot_opening'],
                           promotes_outputs=['mech_angle', 't_1', 'carters_coef'])

        self.add_subsystem(name='equivalent_gap',
                           subsys=GapEquivalentComp(),
                           promotes_inputs=['gap', 'carters_coef', 'k_sat', 't_mag', 'mu_o', 'mu_r'],
                           promotes_outputs=['g_eq', 'g_eq_q'])

        self.add_subsystem(name='torque',
                           subsys=TorqueComp(),
                           promotes_inputs=['B_g', 'n_m', 'n_turns', 'stack_length', 'I', 'rot_or', 'sta_ir'],
                           promotes_outputs=['Tq', 'rot_volume', 'stator_surface_current'])



        # ivc.add_output('I', val=35, units='A', desc='RMS current')
        # ivc.add_output('V', val=385, units='V', desc='RMS dc bus voltage')
        # ivc.add_output('rpm', val=5400, units='RPM', desc='mechanical rpm')
        # ivc.add_output('mu_r', val=1, desc='Relative recoil permeability')
        # ivc.add_output('t_mag', val=0.0044, units='m', desc='Magnet thickness')
        # ivc.add_output('Hc_20', val=1500, units='kA/m', desc='Intrinsic Coercivity at 20 degC')
        # ivc.add_output('Br_20', 1.39, units='T', desc='remnance flux density at 20 degC')
        # ivc.add_output('gap', 0.001, units='m', desc='Air Gap - Mechanical Clearance')
        # ivc.add_output('sta_ir', 0.5, units='m', desc='Inner diameter of the stator')
        # ivc.add_output('n_slots', 1, units=None, desc='Number of slots')  
        # ivc.add_output('l_slot_opening', .002, units='m', desc='Width of the stator slot opening')
        # ivc.add_output('k_sat', 1, units=None, desc='Saturation factor of the magnetic circuit due to the main (linkage) magnetic flux')
        # ivc.add_output('mu_o', 0.4*pi*10**-6, units='H/m', desc='Magnetic Permeability of Free Space')
        # ivc.add_output('n_m', 16, desc='number of magnets')
        # ivc.add_output('n_turns', 16, desc='number of wire turns')
        # ivc.add_output('stack_length', .0345, units='m', desc='stack length')
        # ivc.add_output('rot_or', .025, units='m', desc='rotor outer radius')
