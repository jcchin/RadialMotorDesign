from __future__ import absolute_import
import numpy as np
from math import pi

import openmdao.api as om

from rad_motor.sizing.size_comp import MotorSizeComp, MotorMassComp


class SizeGroup(om.Group):
    def setup(self):

        self.add_subsystem(name='size',
                           subsys=MotorSizeComp(),
                           promotes_inputs=['radius_motor', 'gap', 'rot_or', 'B_g', 'k', 'b_ry', 'n_m', 't_mag',
                                            'b_sy', 'b_t', 'n_slots', 'n_turns', 'I_required', 'k_wb'],
                           promotes_outputs=['J', 'w_ry', 'w_sy', 'w_t', 'sta_ir', 'rot_ir', 's_d', 'slot_area', 'w_slot'])

        self.add_subsystem(name='mass',
                           subsys=MotorMassComp(),
                           promotes_inputs=['rho', 'radius_motor', 'n_slots', 'sta_ir', 'w_t', 'stack_length',
                                            's_d', 'rot_or', 'rot_ir', 't_mag', 'rho_mag', 'L_wire', 'rho_wire', 'r_litz'],
                           promotes_outputs=['mag_mass', 'sta_mass', 'rot_mass', 'wire_mass'])

        adder = om.AddSubtractComp()
        adder.add_equation('mass_total', input_names=['mag_mass', 'sta_mass', 'rot_mass', 'wire_mass'], units='kg')
        self.add_subsystem(name='totalmasscomp', subsys=adder)

        self.connect('mag_mass', 'totalmasscomp.mag_mass')
        self.connect('sta_mass', 'totalmasscomp.sta_mass')
        self.connect('rot_mass', 'totalmasscomp.rot_mass')
        self.connect('wire_mass', 'totalmasscomp.wire_mass')