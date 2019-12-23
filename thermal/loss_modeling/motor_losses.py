from __future__ import absolute_import
import numpy as np
from math import pi
from openmdao.api import Problem, IndepVarComp, ExplicitComponent, ExecComp
from openmdao.api import NewtonSolver, Group, DirectSolver, NonlinearRunOnce, LinearRunOnce, view_model, BalanceComp, ScipyOptimizeDriver


class Losses(ExplicitComponent):
    def setup(self):
        self.add_input('rm', units='rpm', desc='motor speed')
        self.add_input('alpha', 1.27, desc='core loss constant') 
        self.add_input('n_m', desc='number of poles')
        self.add_input('k', .003, desc='windage constant k')
        self.add_input('rho_a', 1.225, units='kg/m**3', desc='air density')
        self.add_input('rot_or', .05, units='m', desc='rotor outer radius')
        self.add_input('rot_ir', .02, units='m', desc='rotor inner radius')
        self.add_input('l_st', units='m', desc='length of stack')
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
        self.add_output('L_total', 15, units='kW', desc='total loss')
        
        self.declare_partials('*','*',method='fd')


    def compute(self, inputs, outputs):
        rm = inputs['rm']
        n_m = inputs['n_m']
        alpha = inputs['alpha']
        f = n_m*rm/120
        rm = inputs['rm']
        k = inputs['k']
        rho_a = inputs['rho_a']
        rot_or = inputs['rot_or']
        rot_ir = inputs['rot_ir']
        l_st = inputs['l_st']
        gap = inputs['gap']
        mu_a = inputs['mu_a']
        omega = rm*(2*pi/60)
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
        outputs['L_airg'] = k*Cfg*pi*rho_a*(omega**3)*(rot_or**4)*l_st
        outputs['L_airf'] = .5*Cfa*rho_a*(omega**3)*((rot_or**5)-(rot_ir**5))
        outputs['L_bear'] = .5*muf_b*D_b*F_b*omega
        outputs['L_total'] = L_airg + L_airf + L_bear + L_emag + L_ewir + L_core + L_res



# ---------------------------------------------------------
# -------------CORE LOSSES---------------------------------
# ---------------------------------------------------------
# Need to have Bpk change with the actual field density
class motor_losses(ExplicitComponent):

    def setup(self):
        self.add_input('K_h', 0.0073790325365744, desc='Hysteresis constant for 0.006in Hiperco-50')
        self.add_input('K_e', 0.00000926301369333214, desc='Eddy constant for 0.006in Hiperco-50')
        self.add_input('f_e', 1000, units='Hz', desc='Electrical frequency')
        self.add_input('K_h_alpha', 1.15293258569149, desc='Hysteresis constant for steinmetz alpha value')
        self.add_input('K_h_beta', 1.72240393990502, desc='Hysteresis constant for steinmetz beta value')
        self.add_input('B_pk', 2.4, units='T', desc='Peak magnetic field in Tesla')

        self.add_output('P_h', units='W', desc='Core hysteresis losses')
        self.add_output('P_e', units='W', desc='Core eddy current losses')
    
    def compute(self,inputs,outputs):
        K_h=inputs['K_h']
        K_e=inputs['K_e']
        K_h_alpha=inputs['K_h_alpha']
        K_h_beta=inputs['K_h_beta']
        B_pk=inputs['B_pk']
        f_e=inputs['f_e']

        outputs['P_h'] = K_h*f_e*B_pk**(K_h_alpha+K_h_beta*B_pk)
        outputs['P_e'] = K_e*f_e**2*B_pk**2

    # def compute_partials(self,inputs,J):





        