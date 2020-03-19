from __future__ import absolute_import
import numpy as np
from math import pi

import openmdao.api as om

from thermal.motor_losses import WindingLossComp, SteinmetzLossComp

motor_loss_data = np.array([
# I:   10          14.4          18.9        23.3          27.8        32.2          36.7        41.1          45.6        50
  [0.008428321, 0.004176335, 0.002550325, 0.001759861, 0.001316224, 0.001042024, 0.000860312, 0.00073333,  0.000640787, 0.000570992],  #RPM=200
  [0.075854887, 0.037587017, 0.022952927, 0.015838752, 0.011846015, 0.009378216, 0.007742812, 0.006599972, 0.005767085, 0.005138924],  #RPM=600
  [0.21070802,  0.10440838,  0.063758132, 0.043996532, 0.032905598, 0.026050599, 0.021507811, 0.018333255, 0.01601968,  0.014274788],  #RPM=1000
  # [0.412987719, 0.204640426, 0.124965938, 0.086233204, 0.064494972, 0.051059174, 0.042155309, 0.035933179, 0.031398573, 0.027978585],#RPM=1400
  [0.682693984, 0.338283153, 0.206576346, 0.142548765, 0.106614138, 0.084403941, 0.069685307, 0.059399745, 0.051903764, 0.046250314],  #RPM=1800
  [1.019826816, 0.505336561, 0.308589357, 0.212943217, 0.159263094, 0.1260849,   0.104097805, 0.088732952, 0.077535252, 0.069089975],  #RPM=2200
  # [1.424386215, 0.705800652, 0.431004969, 0.29741656,  0.222441843, 0.17610205,  0.145392802, 0.123932801, 0.108293037, 0.096497569],#RPM=2600
  [1.896372179, 0.939675424, 0.573823184, 0.395968792, 0.296150382, 0.234455392, 0.193570298, 0.164999291, 0.144177121, 0.128473094],  #RPM=3000
  [2.43578471,  1.206960878, 0.737044001, 0.508599916, 0.380388713, 0.301144925, 0.248630294, 0.211932423, 0.185187502, 0.165016552],  #RPM=3400
  # [3.042623807, 1.507657013, 0.92066742,  0.635309929, 0.475156835, 0.376170651, 0.310572789, 0.264732196, 0.231324181, 0.206127942],#RPM=3800
  [3.716889471, 1.841763831, 1.124693441, 0.776098833, 0.580454749, 0.459532568, 0.379397784, 0.323398611, 0.282587157, 0.251807265],  #RPM=4200
  # [4.458581701, 2.20928133,  1.349122064, 0.930966628, 0.696282454, 0.551230677, 0.455105279, 0.387931667, 0.338976431, 0.302054519],#RPM=4600
  [5.267700498, 2.610209511, 1.593953289, 1.099913312, 0.82263995,  0.651264977, 0.537695273, 0.458331365, 0.400492002, 0.356869706],  #RPM=5000
  [6.14424586,  3.044548373, 1.859187116, 1.282938888, 0.959527238, 0.759635469, 0.627167766, 0.534597704, 0.467133872, 0.416252825]]  #RPM=5400
)



class ThermalGroup(om.Group):

    def setup(self):

        self.add_subsystem('comp', om.ExecComp('I_peak= I*2**0.5'), promotes_inputs=['I'], promotes_outputs=['I_peak'])


        motor_interp = om.MetaModelStructuredComp(method='scipy_slinear')
    
        rpm_data = np.array([200, 600, 1000, 1800, 2200, 3000, 3400, 4200, 5000, 5400])  #  1400, 2600,  3800, 4600
        current_data = np.array([10, 14.4, 18.9, 23.3, 27.8, 32.2, 36.7, 41.1, 45.6, 50])
        
        motor_interp.add_input('rpm', 5400, training_data= rpm_data, units='rpm' )
        motor_interp.add_input('I_peak', 50, training_data=current_data, units='A')
        motor_interp.add_output('AC_power_factor', 0.5, training_data=motor_loss_data)
        self.add_subsystem('ac_power_factor_interp', motor_interp, 
                            promotes_inputs=['rpm', 'I_peak'], promotes_outputs=['AC_power_factor'])

        self.add_subsystem(name='copperloss', 
                           subsys=WindingLossComp(),
                           promotes_inputs=['resistivity_wire', 'stack_length', 'n_slots', 'n_turns', 'T_coeff_cu', 'I',
                                             'T_windings', 'r_strand', 'n_m', 'mu_o', 'mu_r', 'n_strands', 'rpm', 'AC_power_factor'],
                           promotes_outputs=['A_cu', 'f_e', 'r_litz', 'P_dc', 'P_ac', 'P_wire', 'L_wire', 'R_dc', 'skin_depth', 'temp_resistivity'])


        self.add_subsystem(name = 'steinmetzloss',
                           subsys = SteinmetzLossComp(),
                           promotes_inputs=['alpha_stein', 'B_pk', 'f_e', 'beta_stein', 'k_stein', 'motor_mass'],
                           promotes_outputs = ['P_steinmetz'])

