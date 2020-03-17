from __future__ import absolute_import
import numpy as np
from math import pi

import openmdao.api as om

from thermal.ACpowerFactor import ACDC

class WindingLossComp(om.ExplicitComponent):
    # Future: Can include litz wire design from "Simplified Desigh Method for Litz Wire" C.R. Sullivan, R.Y. Zhang


    def setup(self):
        self.add_input('resistivity_wire', 1.724e-8, units='ohm*m', desc='resisitivity of Cu at 20 degC')
        self.add_input('stack_length', 0.035, units='m', desc='axial length of stator')
        self.add_input('n_slots', 20, desc='number of slots')
        self.add_input('n_turns', 12, desc='number of winding turns')
        self.add_input('n_strands', 41, desc='number of strands in litz wire')
        self.add_input('T_coeff_cu', 0.00393, desc='temperature coefficient for copper')
        self.add_input('rpm', 5400, units='rpm', desc='Rotation speed')
        self.add_input('n_m', 20, desc='Number of magnets')
        # self.add_input('T_calc', 150, units='C', desc='average winding temp used for calculations')
        # self.add_input('T_ref_wire', 20, units='C', desc='temperature R_dc is measured at')
        self.add_input('I', 30, units='A', desc='RMS current into motor')
        self.add_input('T_windings', 150, units='C', desc='operating temperature of windings')
        self.add_input('r_strand', 0.0005, units='m', desc='radius of one strand of litz wire')
        self.add_input('mu_o', 0.4*pi*10**-6, units='H/m', desc='permeability of free space')    
        self.add_input('mu_r', 1.0, units='H/m', desc='relative magnetic permeability of ferromagnetic materials') 
        self.add_input('AC_power_factor', 0.5, desc='litz wire AC power factor')

        self.add_output('f_e', 1000, units = 'Hz', desc='electrical frequency')
        self.add_output('P_cu', 500, units='W', desc='copper losses')
        self.add_output('L_wire', 10, units='m', desc='length of wire for one phase')
        self.add_output('R_dc', 1, units='ohm', desc= 'DC resistance')
        self.add_output('skin_depth', 0.001, units='m', desc='skin depth of wire')
        self.add_output('r_litz', 0.002, units='m', desc='radius of whole litz wire')
        self.add_output('P_dc', 100, units='W ', desc= 'Power loss from dc resistance')
        self.add_output('P_ac', 100, units='W ', desc= 'Power loss from ac resistance')
        self.add_output('P_wire', 400, units='W ', desc= 'total power loss from wire')
        self.add_output('temp_resistivity', 1.724e-8, units='ohm*m', desc='temp dependent resistivity')
        self.add_output('A_cu', .005, units='m**2', desc='total area of copper in one slot')

        self.declare_partials('*' , '*', method='fd')
        
    def compute(self, inputs, outputs):
        rpm = inputs['rpm']
        n_m = inputs['n_m']
        mu_o = inputs['mu_o']
        mu_r = inputs['mu_r']
        r_strand = inputs['r_strand']
        T_windings = inputs['T_windings']
        T_coeff_cu = inputs['T_coeff_cu']
        resistivity_wire = inputs['resistivity_wire']
        # T_calc = inputs['T_calc']
        # T_ref_wire = inputs['T_ref_wire']
        I = inputs['I']
        stack_length = inputs['stack_length']
        n_slots = inputs['n_slots']
        n_turns = inputs['n_turns']
        n_strands = inputs['n_strands']

        outputs['f_e']              = n_m / 2 * rpm / 60                                            # Eqn 1.5 "Brushless PM Motor Design" by D. Hansleman
        outputs['r_litz']           = (np.sqrt(n_strands) * 1.154 * r_strand*2)/2                   # New England Wire
        outputs['L_wire']           = (n_slots/3 * n_turns) * (stack_length*2 + .0354)              
        outputs['temp_resistivity'] = (resistivity_wire * (1 + T_coeff_cu*(T_windings-20)))         # Eqn 4.14 "Brushless PM Motor Design" by D. Hansleman
        outputs['R_dc']             = outputs['temp_resistivity'] * outputs['L_wire'] / ((np.pi*(r_strand)**2)*41)
        outputs['skin_depth']       = np.sqrt( outputs['temp_resistivity'] / (np.pi * outputs['f_e'] * mu_r * mu_o) )
        outputs['A_cu']             = n_turns * n_strands * 2 * np.pi * r_strand**2
        outputs['P_dc']             = (I*np.sqrt(2))**2 * (outputs['R_dc']) *3/2
        outputs['P_ac']             = inputs['AC_power_factor'] * outputs['P_dc']
        outputs['P_wire']           = outputs['P_dc'] + outputs['P_ac']


class SteinmetzLossComp(om.ExplicitComponent):

    def setup(self):
        self.add_input('f_e', 1000, units='Hz', desc='Electrical frequency')
        self.add_input('B_pk', 2.05, units='T', desc='Peak magnetic field in Tesla')
        self.add_input('alpha_stein', 1.286, desc='Alpha coefficient for steinmetz, constant')
        self.add_input('beta_stein', 1.76835, desc='Beta coefficient for steinmentz, dependent on freq')  
        self.add_input('k_stein', 0.0044, desc='k constant for steinmentz')
        self.add_output('P_steinmetz', 400, units='W', desc='Simplified steinmetz losses')

        self.declare_partials('*' , '*', method='fd')

    def compute(self, inputs, outputs):
        f_e = inputs['f_e']
        B_pk = inputs['B_pk']
        alpha_stein = inputs['alpha_stein']
        beta_stein = inputs['beta_stein']
        k_stein = inputs['k_stein']

        outputs['P_steinmetz'] = k_stein * f_e**alpha_stein * B_pk**beta_stein

# class LossesComp(ExplicitComponent):
#     def setup(self):
#         self.add_input('rpm', units='rpm', desc='motor speed')
#         self.add_input('alpha', 1.27, desc='core loss constant') 
#         self.add_input('n_m', desc='number of poles')
#         self.add_input('k', .003, desc='windage constant k')
#         self.add_input('rho_a', 1.225, units='kg/m**3', desc='air density')
#         self.add_input('rot_or', .05, units='m', desc='rotor outer radius')
#         self.add_input('rot_ir', .02, units='m', desc='rotor inner radius')
#         self.add_input('stack_length', units='m', desc='length of stack')
#         self.add_input('gap', units='m', desc='air gap') #aka delta
#         self.add_input('mu_a', 1.81e-5, units='(m**2)/s', desc='air dynamic viscosity')
#         self.add_input('muf_b', .3, desc='bearing friction coefficient')
#         self.add_input('D_b', .01, units='m', desc='bearing bore diameter')
#         self.add_input('F_b', 100, units='N', desc='bearing load') #coupled with motor mass
        
#         self.add_output('L_core', units='W', desc='core loss')
#         self.add_output('L_emag', units='W', desc='magnetic eddy loss')
#         self.add_output('L_ewir', units='W', desc='winding eddy loss')
#         self.add_output('L_airg', units='W', desc='air gap windage loss') #air gap
#         self.add_output('L_airf', units='W', desc='axial face windage loss') #axial face
#         self.add_output('L_bear', units='W', desc='bearing loss')
#         self.add_output('L_total', units='kW', desc='total loss')
        
#         self.declare_partials('*','*',method='fd')


#     def compute(self, inputs, outputs):
#         rpm = inputs['rpm']
#         n_m = inputs['n_m']
#         alpha = inputs['alpha']
#         f = n_m*rpm/120
#         k = inputs['k']
#         rho_a = inputs['rho_a']
#         rot_or = inputs['rot_or']
#         rot_ir = inputs['rot_ir']
#         stack_length = inputs['stack_length']
#         gap = inputs['gap']
#         mu_a = inputs['mu_a']
#         omega = rpm*(2*pi/60)
#         Rea = rho_a*omega*(rot_or**2)/mu_a
#         Cfa = .146/(Rea**2)
#         Reg = rho_a*omega*rot_or*gap/mu_a
#         Cfg = .515*((gap**.3)/rot_or)/(Reg**.5)
#         muf_b = inputs['muf_b']
#         D_b = inputs['D_b']
#         F_b = inputs['F_b']

#         outputs['L_core'] = .2157*(f**alpha)
#         outputs['L_emag'] = .0010276*(f**2)
#         outputs['L_ewir'] = .00040681*(f**2)
#         outputs['L_airg'] = k*Cfg*pi*rho_a*(omega**3)*(rot_or**4)*stack_length
#         outputs['L_airf'] = .5*Cfa*rho_a*(omega**3)*((rot_or**5)-(rot_ir**5))
#         outputs['L_bear'] = .5*muf_b*D_b*F_b*omega
#         outputs['L_total'] = outputs['L_airg'] + outputs['L_airf'] + outputs['L_bear'] + outputs['L_emag'] + outputs['L_ewir'] + outputs['L_core']# + outputs['L_res']




# Need to have Bpk change with the actual field density
# class CoreLossComp(ExplicitComponent):

#     def setup(self):
#         self.add_input('K_h',0.0157096642989317, desc='Hysteresis constant for 0.006in Hiperco-50')  #  0.0073790325365744
#         self.add_input('K_e', 8.25786792496325e-7, desc='Eddy constant for 0.006in Hiperco-50')    #  0.00000926301369333214
#         self.add_input('f_e', 1000, units='Hz', desc='Electrical frequency')
#         self.add_input('K_h_alpha', 1.47313466632624, desc='Hysteresis constant for steinmetz alpha value') #  1.15293258569149
#         self.add_input('K_h_beta', 0, desc='Hysteresis constant for steinmetz beta value')           #  1.72240393990502
#         self.add_input('B_pk', 2.05, units='T', desc='Peak magnetic field in Tesla')

#         self.add_output('P_h', units='W/kg', desc='Core hysteresis losses')
#         self.add_output('P_e', units='W/kg', desc='Core eddy current losses')
    
#     def compute(self,inputs,outputs):
#         K_h=inputs['K_h']
#         K_e=inputs['K_e']
#         K_h_alpha=inputs['K_h_alpha']
#         K_h_beta=inputs['K_h_beta']
#         B_pk=inputs['B_pk']
#         f_e=inputs['f_e']

#         outputs['P_h'] = K_h * f_e * B_pk**(K_h_alpha+K_h_beta*B_pk)
#         outputs['P_e'] = 2 * np.pi**2 * K_e * f_e**2 * B_pk**2

    # def compute_partials(self,inputs,J):





