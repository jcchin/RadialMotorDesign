from __future__ import absolute_import
import numpy as np
from math import pi
from openmdao.api import Problem, IndepVarComp, ExplicitComponent, ExecComp
from openmdao.api import NewtonSolver, Group, DirectSolver, NonlinearRunOnce, LinearRunOnce, view_model, BalanceComp, ScipyOptimizeDriver

from thermal.loss_modeling.motor_losses import LossesComp, CoreLossComp

class ThermalGroup(Group):
    def setup(self):
        ivc = IndepVarComp()

        self.add_subsystem(name='losses',
                           subsys=LossesComp(),
                           promotes_inputs=['rho_a', 'alpha', 'mu_a', 'muf_b', 'D_b', 'F_b', 'rpm', 'alpha', 'n_m', 'k', 'rot_or', 'rot_ir', 'stack_length', 'gap',
                                            'mu_a', 'muf_b', 'D_b', 'F_b'],
                           promotes_outputs=['L_core','L_emag', 'L_ewir', 'L_airg', 'L_airf', 'L_bear',
                                             'L_total'])

        self.add_subsystem(name='coreloss',
                           subsys=CoreLossComp(),
                           promotes_inputs=['K_h', 'K_e', 'f_e', 'K_h_alpha', 'K_h_beta', 'B_pk'],
                           promotes_outputs=['P_h', 'P_e'])

        # ivc.add_output('alpha', 1.27, desc='core loss constant') 
        # ivc.add_output('mu_a', 1.81e-5, units='(m**2)/s', desc='air dynamic viscosity')
        # ivc.add_output('muf_b', .3, desc='bearing friction coefficient')
        # ivc.add_output('D_b', .01, units='m', desc='bearing bore diameter')
        # ivc.add_output('F_b', 100, units='N', desc='bearing load') #coupled with motor mass