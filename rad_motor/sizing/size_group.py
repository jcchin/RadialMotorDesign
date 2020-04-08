from __future__ import absolute_import
import numpy as np
from math import pi

import openmdao.api as om

from rad_motor.sizing.size_comp import MotorSizeComp, MotorMassComp, SpecificHeatComp


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
                           promotes_outputs=['mag_mass', 'sta_mass', 'rot_mass', 'wire_mass', 'motor_mass'])

        self.add_subsystem(name='specific_heat',
                           subsys=SpecificHeatComp(),
                           promotes_inputs=['mag_mass', 'sta_mass', 'rot_mass', 'wire_mass', 'motor_mass',
                                            'hiperco_cp', 'copper_cp', 'neo_cp'],
                           promotes_outputs=['mag_cp', 'sta_cp', 'rot_cp', 'wire_cp'])

        adder = om.AddSubtractComp()
        adder.add_equation('cp_motor', input_names=['mag_cp', 'sta_cp', 'rot_cp', 'wire_cp'], units='J/kg/C')
        self.add_subsystem(name='CpMotorComp', subsys=adder, promotes_outputs=['cp_motor'])

        self.connect('mag_cp', 'CpMotorComp.mag_cp')
        self.connect('sta_cp', 'CpMotorComp.sta_cp')
        self.connect('rot_cp', 'CpMotorComp.rot_cp')
        self.connect('wire_cp','CpMotorComp.wire_cp')
