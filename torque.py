from __future__ import absolute_import
import numpy as np
from math import pi, sin, atan, log, e, sqrt
from openmdao.api import Problem, IndepVarComp, ExplicitComponent, ExecComp
from openmdao.api import NewtonSolver, Group, DirectSolver, NonlinearRunOnce, LinearRunOnce, view_model, BalanceComp, ScipyOptimizeDriver

# Reactance:
class Reactance(ExplicitComponent):
    def setup(self):
        self.add_input('m_1', 1, units=None, desc='Number of phases')
        self.add_input('u_0', 0.4*pi*10**-6, units='H/m', desc='Magnetic Permeability of Free Space')  # CONSTANT - Is this the best way to represent a constant?
        self.add_input('f', 1, units='Hz', desc='frequency')
        self.add_input('i', 1, units='A', desc='current')
        self.add_input('N_1', 1, units=None, desc='Number of the stator turns per phase')  # Confirm how this is measured
        self.add_input('k_w1', 1, units=None, desc='Stator winding factor')
        self.add_input('n_m', 1, units=None, desc='Number of poles')  # '2p' in Gieras's book
        self.add_input('tau', 1, units=None, desc='Pole pitch')  # Gieras - pg.134 - (4.27)
        self.add_input('L_i', 1, units='m', desc='Armature stack effective length')
        self.add_input('k_fd', 1, units=None, desc='Form Factor of the Armature Reaction')  # Gieras - pg.192
        self.add_input('k_fq', 1, units=None, desc='Form Factor of the Armature Reaction')  # Gieras - pg.192
        
        # Equivalent Air Gap 
        self.add_input('g_eq', 1, units='m', desc='Equivalent Air Gap') # Gieras - pg.180
        self.add_input('g_eq_q', 1, units='m', desc='Mechanical Clearance in the q-axis')  # Gieras - pg.180

        self.add_input('mag_flux', 1, units='Wb', desc='Magnetic Flux')  # Same as eMag_flux???
        #self.add_input('N_1', 1, units=None, desc='Number of Turns per Phase')
        self.add_output('flux_link', 1, units='Wb', desc='Flux Linkage - Weber-turn')
        self.add_output('L_1', 1, units='H', desc='Leakage Inductance of the armature winding per phase')  # Gieras - pg.204

        self.add_output('pp', 1, units=None, desc='Number of pole pairs')  # 'p' in Gieras's book
        
        self.add_output('X_1', 1, units='ohm', desc='Stator Leakage Reactance')  # Gieras - pg.176
        #self.add_output('X_a', 1, units='ohm', desc='Armature reaction reactance')  # Gieras - pg.176
        self.add_output('X_ad', 1, units='ohm', desc='d-axis armature reaction reactance')  # Gieras - pg.176
        self.add_output('X_aq', 1, units='ohm', desc='q-axis armature reaction reactance')  # Gieras - pg.176
        self.add_output('X_sd', 1, units='ohm', desc='d-axis synchronous reactance')  # Gieras - pg.176 - (5.15)
        self.add_output('X_sq', 1, units='ohm', desc='q-axis synchronous reactance')  # Gieras - pg.176 - (5.15)

    def compute(self, inputs, outputs):
        m_1 = inputs['m_1']
        u_0 = inputs['u_0']
        f = inputs['f']
        i = inputs['i']
        N_1 = inputs['N_1']
        k_w1 = inputs['k_w1']
        n_m = inputs['n_m']
        tau = inputs['tau']
        L_i = inputs['L_i']
        k_fd = inputs['k_fd']
        k_fq = inputs['k_fq']

        g_eq = inputs['g_eq']
        g_eq_q = inputs['g_eq_q']

        mag_flux = inputs['mag_flux']

        outputs['flux_link'] = N_1*mag_flux
        flux_link = outputs['flux_link']
        outputs['L_1'] = flux_link/i
        L_1 = outputs['L_1']

        outputs['pp'] = n_m/2
        pp = outputs['pp']

        outputs['X_1'] = 2*pi*f*L_1
        X_1 = outputs['X_1']
        outputs['X_ad'] = 4*m_1*u_0*f(((N_1*k_w1)**2)/(pi*pp))*(tau*L_i/g_eq)*k_fd
        X_ad = outputs['X_ad']
        outputs['X_aq'] = 4*m_1*u_0*f(((N_1*k_w1)**2)/(pi*pp))*(tau*L_i/g_eq_q)*k_fq
        X_aq = outputs['X_aq']

        outputs['X_sd'] = X_1 + X_ad
        outputs['X_sq'] = X_1 + X_aq

# Equivalent Air Gap Calculations - g' and g'_q
class airGap_eq(ExplicitComponent):
    def setup(self):
        self.add_input('g', 0.001, units='m', desc='Air Gap - Mechanical Clearance')
        self.add_input('k_c', )

# Carter's Coefficient
class k_c(ExplicitComponent):
    def setup(self):
        self.add_input('t_1', 1, units='m', desc='Slot Pitch')  # Gieras - pg.218
        self.add_input('g', 0.001, units='m', desc='Air Gap - Mechanical Clearance')
        self.add_input('D_1in', 1, units='m', desc='Inner diameter of the stator')  # Gieras - pg.217 - Vairable Table
        self.add_input('n_s', 1, units=None, desc='Number of slots')  # 's_1' in Gieras's book
        self.add_input('b_14', 1, units='m', desc='Width of the stator slot opening')  # Gieras
        
        self.add_output('mech_angle', 1, units='rad', desc='Mechanical angle')  # Gieras - pg.563 - (A.28) - LOWER CASE GAMMA
        self.add_output('k_c', 1, units=None, desc='Carters Coefficient')  # Gieras - pg.563 - (A.27)

    def compute(self, inputs, outputs):
        t_1 = inputs['t_1']
        g = inputs['g']
        D_1in = inputs['D_1in']
        n_s = inputs['n_s']
        b_14 = inputs['b_14']

        outputs['mech_angle'] = (4/pi)*(((0.5*b_14*g)*atan(0.5*b_14*g))-(log(sqrt(1+((0.5*b_14*g)**2), e))))
        outputs['k_c'] = (pi*D_1in)/n_s


# First Harmonic of the Air Gap Magnetic Flux Density
class B_mg1(ExplicitComponent):
    def setup(self):
        self.add_input('b_p', 1, units='m', desc='Pole shoe width')  # Gieras - Not Defined
        self.add_input('tau', 1, units=None, desc='Pole pitch')  # Gieras - pg.134 - (4.27)
        self.add_input('B_mg', 2.4, units='T', desc='Magnetic Flux Density under the pole shoe')  # Set to stator max flux density (Hiperco 50) = 2.4T ?  Or calculate.

        self.add_output('pole_arc', 1, units=None, desc='Effective Pole Arc Coefficient')  # Gieras - pg.174 - (4.28) & (5.4)
        self.add_output('B_mg1', 1, units='T', desc='Air Gap Magnetic Flux Density')

    def compute(self, inputs, outputs):
        b_p = inputs['b_p']
        tau = inputs['tau']
        B_mg = inputs['B_mg']

        outputs['pole_arc'] = b_p/tau
        pole_arc = outputs['pole_arc']
        outputs['B_mg1'] = (4/pi)*B_mg*sin(0.5*pole_arc*pi)

# Excitation Magnetic Flux
class eMag_flux(ExplicitComponent):
    def setup(self):
        self.add_input('L_i', 1, units='m', desc='Armature stack effective length')
        self.add_input('B_mg1', 1, units='T', desc='Air Gap Magnetic Flux Density')  # Should we calculate or insert value?
        self.add_input('mot_or', .075, units='m', desc='motor outer radius')
        self.add_input('n_m', 1, units=None, desc='Number of poles')  # '2p' in Gieras's book
        
        self.add_output('tau', 1, units='m', desc='Pole pitch')  # Gieras - pg.134 - (4.27)
        self.add_output('eMag_flux', 1, units='Wb', desc='Excitation Magnetic Flux')

    def compute(self, inputs, outputs):
        L_i = inputs['L_i']
        B_mg1 = inputs['B_mg1']
        mot_or = inputs['mot_or']
        n_m = inputs['n_m']

        outputs['tau'] = (2*pi*mot_or)/n_m
        tau = outputs['tau']
        outputs['eMag_flux'] = (2/pi)*tau*L_i*B_mg1

# Stator Winding Factor
class k_w1(ExplicitComponent):
    def setup(self):
        self.add_input('w_sl', 1, units=None, desc='Coil span measured in number of slots')  # Do we know how to measure? - Create picture showing how it is measured?
        self.add_input('m_1', 1, units=None, desc='Number of phases')
        self.add_input('n_m', 1, units=None, desc='Number of poles')  # '2p' in Gieras's book
        self.add_input('n_s', 1, units=None, desc='Number of slots')  # 's_1' in Gieras's book

        self.add_output('pps', 1, units='rad', desc='Poles Per Slot - Angular displacement between adjacent slots in electrical degrees')
        self.add_output('q_1', 1, units=None, desc='Number of slots per pole per phase')
        self.add_output('Q_1', 1, units=None, desc='Number of slots per pole')
        self.add_output('k_d1', 1, units=None, desc='Distribution factor')
        self.add_output('k_p1', 1, units=None, desc='Pitch Factor')
        self.add_output('k_w1', 1, units=None, desc='Stator winding factor')

    def compute(self, inputs, outputs):
        w_sl = inputs['w_sl']
        m_1 = inputs['m_1']
        n_m = inputs['n_m']
        n_s = inputs['n_s']

        outputs['pps'] = (pi*n_m)/n_s
        outputs['q_1'] = n_s/(n_m*m_1)
        outputs['Q_1'] = n_s/n_m

        pps = outputs['pps']
        q_1 = outputs['q_1']
        Q_1 = outputs['Q_1']
        outputs['k_d1'] = (sin(0.5*q_1*pps))/(q_1*sin(0.5*pps))
        outputs['k_p1'] = sin(0.5*pi*w_sl/Q_1)
        k_d1 = outputs['k_d1']
        k_p1 = outputs['k_p1']
        outputs['k_w1'] = k_d1*k_p1

# Frequency:
class Frequency(ExplicitComponent):
    def setup(self):
        self.add_input('rm', 1, units='rpm', desc='motor speed')  # "n_s" in Gieras's book
        self.add_input('p_p', 1, units=None, desc='Number of pole pairs')

        self.add_output('f', 1, units='Hz', desc='frequency')

    def compute(self, inputs, outputs):
        rm = inputs['rm']
        p_p = inputs['p_p']

        outputs['f'] = rm*p_p

# EMF
class E_f(ExplicitComponent):
    def setup(self):
        self.add_input('rm', 1, units='rpm', desc='motor speed')  # "n_s" in Gieras's book
        self.add_input('p_p', 1, units=None, desc='Number of pole pairs')
        self.add_input('N_1', 1, units=None, desc='Number of the stator turns per phase')  # How do we get this?
        self.add_input('k_w1', 1, units=None, desc='the stator winding coefficient')  # Computed in the "k_w1" class TODO: Connect k_w1 output to here
        self.add_input('eMag_flux', 1, units='Wb', desc='Excitation Magnetic Flux')  # What value to use?  Or does it need to be calculated?

        self.add_output('f', 1, units='Hz', desc='frequency')
        self.add_output('E_f', 1, units='V', desc='EMF - the no-load RMS Voltage induced in one phase of the stator winding')

    def compute(self, inputs, outputs):
        rm = inputs['rm']
        p_p = inputs['p_p']
        N_1 = inputs['N_1']
        k_w1 = inputs['k_w1']
        b_mag = inputs['b_mag']

        outputs['f'] = rm*p_p
        f = outputs['f']

        outputs['E_f'] = pi*(2**0.5)*f*N_1*k_w1*b_mag

# Torque
class torque(ExplicitComponent):
    def setup(self):
        self.add_input('m_1', 1, units=None, desc='number of phases')
        self.add_input('rm', 1, units='rpm', desc='motor speed')  # "n_s" in Gieras's book
        self.add_input('V_1', 1, units='V', desc='stator voltage')  # Confirm that this is the same as the bus volage
        self.add_input('E_f', 1, units='V', desc='EMF - the no-load RMS Voltage induced in one phase of the stator winding')
        self.add_input('X_sd', 1, units='ohm', desc='d-axis synchronous reactance')
        self.add_input('X_sq', 1, units='ohm', desc='q-axis synchronous reactance')
        self.add_input('delta', 1, units='rad', desc='Power (Load) Angle - The angle between V-1 and E_f')  #TODO: Check units and calculation
        
        self.add_output('p_elm', 1, units='W', desc='Power - Electromagnetic')
        self.add_output('tq', 1, units='N*m', desc='Torque - Electromagnetic')

    def compute(self, inputs, outputs):
        m_1 = inputs['m_1']
        rm = inputs['rm']
        V_1 = inputs['V_1']
        E_f = inputs['E_f']
        X_sd = inputs['X_sd']
        X_sq = inputs['X_sq']
        delta = inputs['delta']

        #outputs['tq'] = (m_1/(2*pi*rm))((V_1*E_f*sin(delta))+(((V_1**2)/2)((1/X_sq)-(1/X_sd))*sin(2*delta)))
        outputs['p_elm'] = (m_1)((V_1*E_f*sin(delta))+(((V_1**2)/2)((1/X_sq)-(1/X_sd))*sin(2*delta)))
        p_elm = outputs['p_elm']
        outputs['tq'] = p_elm/(2*pi*rm)
