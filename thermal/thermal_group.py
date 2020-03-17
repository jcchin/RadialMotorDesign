from __future__ import absolute_import
import numpy as np
from math import pi

import openmdao.api as om

from thermal.motor_losses import WindingLossComp, SteinmetzLossComp


class ThermalGroup(om.Group):
    def setup(self):

        # self.add_subsystem(name='losses',
        #                    subsys=LossesComp(),
        #                    promotes_inputs=['rho_a', 'alpha', 'mu_a', 'muf_b', 'D_b', 'F_b', 'rpm', 'alpha', 'n_m', 'k', 'rot_or', 'rot_ir', 'stack_length', 'gap',
        #                                     'mu_a', 'muf_b', 'D_b', 'F_b'],
        #                    promotes_outputs=['L_core','L_emag', 'L_ewir', 'L_airg', 'L_airf', 'L_bear',
        #                                      'L_total'])

        # self.add_subsystem(name='coreloss',
        #                    subsys=CoreLossComp(),
        #                    promotes_inputs=['K_h', 'K_e', 'f_e', 'K_h_alpha', 'K_h_beta', 'B_pk'],
        #                    promotes_outputs=['P_h', 'P_e'])

        self.add_subsystem(name='copperloss',
                           subsys=WindingLossComp(),
                           promotes_inputs=['resistivity_wire', 'stack_length', 'n_slots', 'n_turns', 'T_coeff_cu', 'I',
                                             'T_windings', 'r_strand', 'n_m', 'mu_o', 'mu_r', 'n_strands', 'rpm'],
                           promotes_outputs=['A_cu', 'f_e', 'r_litz', 'P_dc', 'P_ac', 'P_wire', 'L_wire', 'R_dc', 'skin_depth', 'temp_resistivity'])

        self.add_subsystem(name = 'steinmetzloss',
                           subsys = SteinmetzLossComp(),
                           promotes_inputs=['alpha_stein', 'B_pk', 'f_e', 'beta_stein', 'k_stein'],
                           promotes_outputs = ['P_steinmetz'])

        # ivc.add_output('alpha', 1.27, desc='core loss constant') 
        # ivc.add_output('mu_a', 1.81e-5, units='(m**2)/s', desc='air dynamic viscosity')
        # ivc.add_output('muf_b', .3, desc='bearing friction coefficient')
        # ivc.add_output('D_b', .01, units='m', desc='bearing bore diameter')
        # ivc.add_output('F_b', 100, units='N', desc='bearing load') #coupled with motor mass