# Temp of magnet is assumed steady state through all analysis.
# Carters is calculated only with dB, appropriate for n48H magnet

from __future__ import absolute_import
import numpy as np
from math import pi

import openmdao.api as om

class CartersComp(om.ExplicitComponent):

    def setup(self):
        self.add_input('gap', 0.001, units='m', desc='Air Gap - Mechanical Clearance')
        self.add_input('w_slot', .015, units='m', desc='width of one slot')
        self.add_input('w_t', .0045, units='m', desc='tooth width')
        self.add_input('t_mag', .0044, units='m', desc='radial thickness of magnet')
        self.add_input('Br_20', 1.39, units='T', desc='remnance flux density at 20 degC')
        self.add_input('T_mag', 100, units='C', desc='operating temperature of magnet')
        self.add_input('T_coef_rem_mag', -0.12,  desc=' Temperature coefficient of the remnance flux density for N48H magnets')
        
        self.add_output('Br', 1, units = 'T', desc='temp dependent renmance flux density of an N48H magnet')
        self.add_output('carters_coef', 1,  desc='How much the air gap must be increased to account for slots')  # Gieras - pg.563 - (A.27)

        self.declare_partials('Br', ['Br_20', 'T_coef_rem_mag', 'T_mag'])
        # self.declare_partials('carters_coef', ['w_slot', 'w_t', 'gap', 't_mag', 'Br_20', 'T_coef_rem_mag', 'T_mag'])

    def compute(self, inputs, outputs):
        g = inputs['gap']
        w_slot = inputs['w_slot']
        w_t = inputs['w_t']
        t_mag = inputs['t_mag']
        Br_20 = inputs['Br_20']
        T_mag = inputs['T_mag']
        T_coef_rem_mag = inputs['T_coef_rem_mag']

        outputs['Br']  = Br_20*(1+T_coef_rem_mag/100 * (T_mag-20)) 
        outputs['carters_coef'] =  2 # (1 - w_slot/(w_slot + w_t)) + ((4*(g+t_mag/outputs['Br'])/(np.pi*(w_slot + w_t))) * np.log(1 + (np.pi*w_slot/(4*(g+t_mag/outputs['Br'])))))**-1 

    def compute_partials(self, inputs, J):
        g = inputs['gap']
        w_slot = inputs['w_slot']
        w_t = inputs['w_t']
        t_mag = inputs['t_mag']
        Br_20 = inputs['Br_20']
        T_coef_rem_mag = inputs['T_coef_rem_mag']
        T_mag = inputs['T_mag']

        Br  = Br_20*(1+T_coef_rem_mag/100 * (T_mag-20)) 
        # Remnant flux density
        J['Br', 'Br_20'] = (1+(T_coef_rem_mag)/100 * (T_mag-20))  
        J['Br', 'T_coef_rem_mag'] = Br_20/100*(T_mag-20)
        J['Br', 'T_mag'] = Br_20*T_coef_rem_mag/100

        # # Carters Coefficient
        
        # J['carters_coef', 'w_slot'] = (-1*(w_slot + w_t) + w_slot)/(w_slot + w_t)**2 - \
        #                                 ((4*(g+t_mag/Br)/(np.pi*(w_slot + w_t))) * np.log(1 + (np.pi*w_slot/(4*(g+t_mag/Br)))))**-2 * \
        #                                 (((-np.pi*4*(g+t_mag/Br))/(np.pi*(w_slot+w_t))**2) * np.log(1 + (np.pi*w_slot/(4*(g+t_mag/Br)))) \
        #                                 + ((4*(g+t_mag/Br)/(np.pi*(w_slot + w_t)))*(1/(1 + (np.pi*w_slot/(4*(g+t_mag/Br)))))*(np.pi/(4*(g+t_mag/Br)))))
        
        # J['carters_coef', 'w_t'] = w_slot/(w_slot + w_t)**2 - ((4*(g+t_mag/Br)/(np.pi*(w_slot + w_t))) * np.log(1 + (np.pi*w_slot/(4*(g+t_mag/Br)))))**-2 * \
        #                             (((-np.pi*4*(g+t_mag/Br))/(np.pi*(w_slot+w_t))**2) * np.log(1 + (np.pi*w_slot/(4*(g+t_mag/Br)))))
        
        # J['carters_coef', 'gap'] = -1 * \
        #                             ((4*(g+t_mag/Br)/(np.pi*(w_slot + w_t))) * np.log(1 + (np.pi*w_slot/(4*(g+t_mag/Br)))))**-2 \
        #                             * ((4/(np.pi*(w_slot + w_t))) * np.log(1 + (np.pi*w_slot/(4*(g+t_mag/Br)))) +
        #                                 1/(1 + (np.pi*w_slot/(4*(g+t_mag/Br)))) * (-np.pi*w_slot*4/(4*(g+t_mag/Br))**2) * \
        #                                 4*(g+t_mag/Br)/(np.pi*(w_slot + w_t)))

        # J['carters_coef', 't_mag'] = -1 * ((4*(g+t_mag/Br)/(np.pi*(w_slot + w_t))) * np.log(1 + (np.pi*w_slot/(4*(g+t_mag/Br)))))**-2 \
        #                             * ((4/(Br*np.pi*(w_slot + w_t)))*np.log(1 + (np.pi*w_slot/(4*(g+t_mag/Br)))) \
        #                             + (4*(g+t_mag/Br)/(np.pi*(w_slot + w_t)))*(1/(1 + (np.pi*w_slot/(4*(g+t_mag/Br)))))*((-4/Br)*np.pi*w_slot/(4*(g+t_mag/Br))**2))
        
        # dBr__dBr_20 = (1+T_coef_rem_mag/100 * (T_mag-20))
        # dBe1__dBr_20 = -(4*t_mag/(np.pi*(w_slot+w_t)*Br**2)) * dBr__dBr_20
        # dBe2__dBr_20 = (4*t_mag*Br**-2*dBr__dBr_20 * np.pi*w_slot)/(4*g + 4*t_mag/Br)**2
        # J['carters_coef', 'Br_20'] = -1 * ((4*(g+t_mag/Br)/(np.pi*(w_slot + w_t))) * np.log(1 + (np.pi*w_slot/(4*(g+t_mag/Br)))))**-2 \
        #                              * ((dBe1__dBr_20 * np.log(1 + (np.pi*w_slot/(4*(g+t_mag/Br))))) + (((4*(g+t_mag/Br)/(np.pi*(w_slot + w_t))) * (1/(1 + (np.pi*w_slot/(4*(g+t_mag/Br))))))
        #                                 * dBe2__dBr_20))


        # dBr__dT_coef_rem_mag = (T_mag-20)*Br_20/100
        # dBe1__dT_coef_rem_mag = -(4*t_mag/(np.pi*(w_slot+w_t)*Br**2)) * dBr__dT_coef_rem_mag
        # dBe2__dT_coef_rem_mag = (4*t_mag*Br**-2*dBr__dT_coef_rem_mag * np.pi*w_slot)/(4*g + 4*t_mag/Br)**2
        # J['carters_coef', 'T_coef_rem_mag'] = -1 * ((4*(g+t_mag/Br)/(np.pi*(w_slot + w_t))) * np.log(1 + (np.pi*w_slot/(4*(g+t_mag/Br)))))**-2 \
        #                              * ((dBe1__dT_coef_rem_mag * np.log(1 + (np.pi*w_slot/(4*(g+t_mag/Br))))) + (((4*(g+t_mag/Br)/(np.pi*(w_slot + w_t))) * (1/(1 + (np.pi*w_slot/(4*(g+t_mag/Br))))))
        #                                 * dBe2__dT_coef_rem_mag))
        
        # dBr__dT_mag = Br_20 * T_coef_rem_mag/100
        # dBe1__dT_mag = -(4*t_mag/(np.pi*(w_slot+w_t)*Br**2)) * dBr__dT_mag
        # dBe2__dT_mag = (4*t_mag*Br**-2*dBr__dT_mag * np.pi*w_slot)/(4*g + 4*t_mag/Br)**2
        # J['carters_coef', 'T_mag'] = -1 * ((4*(g+t_mag/Br)/(np.pi*(w_slot + w_t))) * np.log(1 + (np.pi*w_slot/(4*(g+t_mag/Br)))))**-2 \
        #                              * ((dBe1__dT_mag * np.log(1 + (np.pi*w_slot/(4*(g+t_mag/Br))))) + (((4*(g+t_mag/Br)/(np.pi*(w_slot + w_t))) * (1/(1 + (np.pi*w_slot/(4*(g+t_mag/Br))))))
        #                                 * dBe2__dT_mag))


class GapEquivalentComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('gap', 0.001, units='m', desc='Air Gap - Mechanical Clearance')
        self.add_input('carters_coef', 2,  desc='Carters Coefficient')  # Gieras - pg.563 - (A.27)
        self.add_input('k_sat', 1,  desc='Saturation factor of the magnetic circuit due to the main (linkage) magnetic flux')  # Gieras - pg.73 - (2.48) - Typically ~1

        self.add_output('g_eq', .001, units='m', desc='Equivalent aig gap')  # Gieras - pg.180

        self.declare_partials('g_eq', ['gap', 'carters_coef', 'k_sat'])
        # self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        gap = inputs['gap']
        carters_coef = inputs['carters_coef']
        k_sat = inputs['k_sat']

        outputs['g_eq'] = gap*carters_coef*k_sat 

        # outputs['g_eq'] = .002

    def compute_partials(self, inputs, J):
        gap = inputs['gap']
        carters_coef = inputs['carters_coef']
        k_sat = inputs['k_sat']

        J['g_eq', 'gap'] = carters_coef*k_sat
        J['g_eq', 'carters_coef'] = gap*k_sat
        J['g_eq', 'k_sat'] = gap*carters_coef


class GapFieldsComp(om.ExplicitComponent):

  def setup(self):
    self.add_input('mu_r', 1.04, units='H/m', desc='relative magnetic permeability of ferromagnetic materials')
    self.add_input('g_eq', .001, units='m', desc='air gap')
    self.add_input('t_mag', 0.0045, units='m', desc='magnet height')
    self.add_input('Br', 1, units = 'T', desc='temp dependent renmance flux density of an N48H magnet')

    self.add_output('B_g', 1.5, units='T', desc='air gap flux density')

    self.declare_partials('B_g', ['Br', 'mu_r', 'g_eq', 't_mag'])

  def compute(self, inputs, outputs):
    Br = inputs['Br']
    mu_r=inputs['mu_r']
    g_eq=inputs['g_eq']
    t_mag=inputs['t_mag']

    outputs['B_g'] = Br/(1+mu_r*(g_eq/t_mag))                   # neglecting leakage flux and fringing, magnetic voltag drop in steel (eqn2.14 Gieres PMSM)

  def compute_partials(self, inputs, J):
    Br = inputs['Br']
    mu_r=inputs['mu_r']
    g_eq=inputs['g_eq']
    t_mag=inputs['t_mag']

    J['B_g', 'Br'] = (1/(1+mu_r*g_eq/t_mag)) 
    J['B_g', 'mu_r'] = -Br*g_eq*t_mag/((g_eq*mu_r+t_mag)**2)
    J['B_g', 'g_eq'] = -Br*(mu_r/t_mag)/(1+mu_r*(g_eq/t_mag))**2
    J['B_g', 't_mag'] = Br * mu_r*g_eq*t_mag**-2 / (1+mu_r*(g_eq/t_mag))**2


