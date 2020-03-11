from __future__ import absolute_import
import numpy as np
from math import pi
from openmdao.api import Problem, IndepVarComp, ExplicitComponent, ExecComp
from openmdao.api import NewtonSolver, Group, DirectSolver, NonlinearRunOnce, LinearRunOnce, view_model, BalanceComp, ScipyOptimizeDriver

from sizing.size_comp import MotorSizeComp, MotorMassComp


class SizeGroup(Group):
    def setup(self):
        ivc = IndepVarComp()

        self.add_subsystem(name='size',
                           subsys=MotorSizeComp(),
                           promotes_inputs=['radius_motor', 'gap', 'rot_or', 'B_g', 'k', 'b_ry', 'n_m', 't_mag',
                                            'b_sy', 'b_t', 'n_slots', 'n_turns', 'I', 'k_wb'],
                           promotes_outputs=['J', 'w_ry', 'w_sy', 'w_t', 'sta_ir', 'rot_ir', 's_d', 'slot_area'])

        self.add_subsystem(name='mass',
                           subsys=MotorMassComp(),
                           promotes_inputs=['rho', 'radius_motor', 'n_slots', 'sta_ir', 'w_t', 'stack_length',
                                            's_d', 'rot_or', 'rot_ir', 't_mag', 'rho_mag'],
                           promotes_outputs=['mag_mass', 'sta_mass', 'rot_mass'])