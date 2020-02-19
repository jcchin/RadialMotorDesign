from __future__ import absolute_import
import numpy as np
from math import pi
from openmdao.api import Problem, IndepVarComp, ExplicitComponent, ExecComp
from openmdao.api import NewtonSolver, Group, DirectSolver, NonlinearRunOnce, LinearRunOnce, view_model, BalanceComp, ScipyOptimizeDriver


class LossesComp(ExplicitComponent):
    def setup(self):
        self.add_input('rpm', units='rpm', desc='motor speed')
        self.add_input('alpha', 1.27, desc='core loss constant') 
        self.add_input('n_m', desc='number of poles')
        self.add_input('k', .003, desc='windage constant k')
        self.add_input('rho_a', 1.225, units='kg/m**3', desc='air density')
        self.add_input('rot_or', .05, units='m', desc='rotor outer radius')
        self.add_input('rot_ir', .02, units='m', desc='rotor inner radius')
        self.add_input('stack_length', units='m', desc='length of stack')
        self.add_input('gap', units='m', desc='air gap') #aka delta
        self.add_input('mu_a', 1.81e-5, units='(m**2)/s', desc='air dynamic viscosity')
        self.add_input('muf_b', .3, desc='bearing friction coefficient')
        self.add_input('D_b', .01, units='m', desc='bearing bore diameter')
        self.add_input('F_b', 100, units='N', desc='bearing load') #coupled with motor mass
        
        self.add_output('L_core', units='W', desc='core loss')
        self.add_output('L_emag', units='W', desc='magnetic eddy loss')
        self.add_output('L_ewir', units='W', desc='winding eddy loss')
        self.add_output('L_airg', units='W', desc='air gap windage loss') #air gap
        self.add_output('L_airf', units='W', desc='axial face windage loss') #axial face
        self.add_output('L_bear', units='W', desc='bearing loss')
        self.add_output('L_total', units='kW', desc='total loss')
        
        self.declare_partials('*','*',method='fd')


    def compute(self, inputs, outputs):
        rpm = inputs['rpm']
        n_m = inputs['n_m']
        alpha = inputs['alpha']
        f = n_m*rpm/120
        k = inputs['k']
        rho_a = inputs['rho_a']
        rot_or = inputs['rot_or']
        rot_ir = inputs['rot_ir']
        stack_length = inputs['stack_length']
        gap = inputs['gap']
        mu_a = inputs['mu_a']
        omega = rpm*(2*pi/60)
        Rea = rho_a*omega*(rot_or**2)/mu_a
        Cfa = .146/(Rea**2)
        Reg = rho_a*omega*rot_or*gap/mu_a
        Cfg = .515*((gap**.3)/rot_or)/(Reg**.5)
        muf_b = inputs['muf_b']
        D_b = inputs['D_b']
        F_b = inputs['F_b']

        outputs['L_core'] = .2157*(f**alpha)
        outputs['L_emag'] = .0010276*(f**2)
        outputs['L_ewir'] = .00040681*(f**2)
        outputs['L_airg'] = k*Cfg*pi*rho_a*(omega**3)*(rot_or**4)*stack_length
        outputs['L_airf'] = .5*Cfa*rho_a*(omega**3)*((rot_or**5)-(rot_ir**5))
        outputs['L_bear'] = .5*muf_b*D_b*F_b*omega
        outputs['L_total'] = outputs['L_airg'] + outputs['L_airf'] + outputs['L_bear'] + outputs['L_emag'] + outputs['L_ewir'] + outputs['L_core']# + outputs['L_res']




# Need to have Bpk change with the actual field density
class CoreLossComp(ExplicitComponent):

    def setup(self):
        self.add_input('K_h',0.0157096642989317, desc='Hysteresis constant for 0.006in Hiperco-50')  #  0.0073790325365744
        self.add_input('K_e', 8.25786792496325e-7, desc='Eddy constant for 0.006in Hiperco-50')    #  0.00000926301369333214
        self.add_input('f_e', 1000, units='Hz', desc='Electrical frequency')
        self.add_input('K_h_alpha', 1.47313466632624, desc='Hysteresis constant for steinmetz alpha value') #  1.15293258569149
        self.add_input('K_h_beta', 0, desc='Hysteresis constant for steinmetz beta value')           #  1.72240393990502
        self.add_input('B_pk', 2.05, units='T', desc='Peak magnetic field in Tesla')

        self.add_output('P_h', units='W/kg', desc='Core hysteresis losses')
        self.add_output('P_e', units='W/kg', desc='Core eddy current losses')
    
    def compute(self,inputs,outputs):
        K_h=inputs['K_h']
        K_e=inputs['K_e']
        K_h_alpha=inputs['K_h_alpha']
        K_h_beta=inputs['K_h_beta']
        B_pk=inputs['B_pk']
        f_e=inputs['f_e']

        outputs['P_h'] = K_h * f_e * B_pk**(K_h_alpha+K_h_beta*B_pk)
        outputs['P_e'] = 2 * np.pi**2 * K_e * f_e**2 * B_pk**2

    # def compute_partials(self,inputs,J):


class WindingLossComp(ExplicitComponent):

    def setup(self):
        self.add_input('resistivity_wire', 0.3, units='ohm',desc='density of the wire material')
        self.add_input('stack_length', 0.035, units='m', desc='axial length of stator')
        self.add_input('n_slots', 20, desc='number of slots')
        self.add_input('n_turns', 12, desc='number of winding turns')
        self.add_input('T_coeff_cu', 1, desc='temperature coefficient for copper')
        self.add_input('T_calc', 150, units='C', desc='average winding temp used for calculations')
        self.add_input('T_ref_wire', 20, units='C', desc='temperature R_dc is measured at')
        self.add_input('I', 30, units='A', desc='current into motor')
        self.add_input('R_dc', 1, desc = 'NEEDS TO BE DEFINED')

        self.add_output('R_ph', 0.5, units='ohm', desc='resistance in each phase')
        self.add_output('P_cu', 500, units='W', desc='copper losses')
        self.add_output('L_wire', 10, units='m', desc='length of wire for one phase')
        self.add_output('A_wire', .001, units='m**2', desc='cross sectional area of wire')
        self.add_output('R_dc_litz', 1, desc= 'NEEDS_UPDATE')


    def compute(self, inputs, outputs):
        R_dc=inputs['R_dc']
        T_coeff_cu=inputs['T_coeff_cu']
        T_calc=inputs['T_calc']
        T_ref_wire=inputs['T_ref_wire']
        I=inputs['I']
        stack_length = inputs['stack_length']
        n_slots = inputs['n_slots']
        n_turns = inputs['n_turns']



        outputs['L_wire'] = (stack_length * n_slots/3 * n_turns) + 0.010
        outputs['R_dc_litz'] = ((40 * 1.015**1 * 1.025**1) / 48)
        outputs['A_wire'] = .01
        outputs['R_ph'] = ((40 * 1.015**1 * 1.025**1) / 48) * (1+T_coeff_cu*(T_calc - T_ref_wire))
        outputs['P_cu'] = (3/2) * I**2 * outputs['R_ph']

class SteinmetzLossComp(ExplicitComponent):
    def setup(self):
        self.add_input('f_e', 1000, units='Hz', desc='Electrical frequency')
        self.add_input('B_pk', 2.05, units='T', desc='Peak magnetic field in Tesla')
        self.add_input('alpha_stein', 1.286, desc='Alpha coefficient for steinmetz, constant')
        self.add_input('beta_stein', 1.76835, desc='Beta coefficient for steinmentz, dependent on freq')    # needs a looup table as fun(freq)
        self.add_input('k_stein', 0.0044, desc='k constant for steinmentz')
        self.add_output('P_steinmetz', 400, desc='Simplified steinmetz losses')


    def compute(self, inputs, outputs):
        f_e = inputs['f_e']
        B_pk = inputs['B_pk']
        alpha_stein = inputs['alpha_stein']
        beta_stein = inputs['beta_stein']
        k_stein = inputs['k_stein']

        outputs['P_steinmetz'] = k_stein * f_e**alpha_stein * B_pk**beta_stein
