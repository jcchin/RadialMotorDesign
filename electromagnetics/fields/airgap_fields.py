from __future__ import absolute_import
import numpy as np
from math import pi
from openmdao.api import Problem, IndepVarComp, ExplicitComponent, ExecComp
from openmdao.api import NewtonSolver, Group, DirectSolver, NonlinearRunOnce, LinearRunOnce, view_model, BalanceComp, ScipyOptimizeDriver



class airgap_field_intsity(ExplicitComponent):

  def setup(self):
    self.add_input('B_g', 1.5, units='T', desc='air gap flux density')
    self.add_input('H_c', 1500000, units='A/m', desc='Intrinsic Coercivity')
    self.add_input('B_r', 1.39, units='T', desc='remnance flux density')

    self.add_output('H_g', units='A/m', desc='air gap field intensity')

    self.declare_partials('*','*', method='fd')

  def compute(self, inputs, outputs):
    B_g=inputs['B_g']
    H_c=inputs['H_c']
    B_r=inputs['B_r']

    outputs['H_g'] = H_c*(B_g/B_r)    # Only valid at 20 deg C, and neglects voltage drop across air gap, input eqn 2.17


class carters_coefficient(ExplicitComponent):
    def setup(self):
        self.add_input('gap', 0.001, units='m', desc='Air Gap - Mechanical Clearance')
        self.add_input('sta_ir', 1, units='m', desc='Inner diameter of the stator')  # Gieras - pg.217 - Vairable Table
        self.add_input('n_slots', 1, units=None, desc='Number of slots')  # 's_1' in Gieras's book
        self.add_input('l_slot_opening', .002, units='m', desc='Width of the stator slot opening')  # Gieras
        
        # 'mech_angle' in calculated in rad
        self.add_output('mech_angle', 1, units='rad', desc='Mechanical angle')  # Gieras - pg.563 - (A.28) - LOWER CASE GAMMA
        self.add_output('t_1', 1, units='m', desc='Slot Pitch')  # Gieras - pg.218
        self.add_output('carters_coef', 1, units=None, desc='Carters Coefficient')  # Gieras - pg.563 - (A.27)

    def compute(self, inputs, outputs):
        g = inputs['gap']
        sta_ir = inputs['sta_ir'] * 2
        n_slots = inputs['n_slots']
        l_slot_opening = inputs['l_slot_opening']

        outputs['mech_angle'] = (4/np.pi)*(((0.5*l_slot_opening/g)*np.arctan(0.5*l_slot_opening/g))-(np.log(np.sqrt(1+((0.5*l_slot_opening/g)**2)))))
        outputs['t_1'] = (pi*sta_ir)/n_slots

        t_1 = outputs['t_1']
        mech_angle = outputs['mech_angle']

        outputs['carters_coef'] = t_1/(t_1 - (mech_angle*g))


class airgap_eq(ExplicitComponent):
    def setup(self):
        self.add_input('gap', 0.001, units='m', desc='Air Gap - Mechanical Clearance')
        self.add_input('carters_coef', 1, units=None, desc='Carters Coefficient')  # Gieras - pg.563 - (A.27)
        self.add_input('k_sat', 1, units=None, desc='Saturation factor of the magnetic circuit due to the main (linkage) magnetic flux')  # Gieras - pg.73 - (2.48) - Typically ~1
        self.add_input('t_mag', 0.0044, units='m', desc='Magnet thickness')  # 'h_m' in Gieras's book
        self.add_input('mu_o', 0.4*pi*10**-6, units='H/m', desc='Magnetic Permeability of Free Space')  #CONSTANT
        self.add_input('mu_r', 1, units=None, desc='Relative recoil permeability')  # Gieras - pg.48 - (2.5)

        self.add_output('g_eq', 1, units='m', desc='Equivalent aig gap')  # Gieras - pg.180
        self.add_output('g_eq_q', 1, units='m', desc='Equivalent air gap q-axis')  # Gieras - pg.180

    def compute(self, inputs, outputs):
        g = inputs['gap']
        carters_coef = inputs['carters_coef']
        k_sat = inputs['k_sat']
        t_mag = inputs['t_mag']
        mu_o = inputs['mu_o']
        mu_r = inputs['mu_r']

        outputs['g_eq'] = g*carters_coef*k_sat #+ (t_mag/mu_r)
        outputs['g_eq_q'] = g*carters_coef*k_sat


class gap_flux_densities(ExplicitComponent):

  def setup(self):
    self.add_input('B_r', 1.39, units='T', desc='remnance flux density')
    self.add_input('mu_r', 1.04, units='H/m', desc='relative magnetic permeability of ferromagnetic materials')
    self.add_input('g_eq', .001, units='m', desc='air gap')
    self.add_input('t_mag', 0.0045, units='m', desc='magnet height')

    self.add_output('B_g', units='T', desc='air gap flux density')

    self.declare_partials('*','*', method='fd')

  def compute(self, inputs, outputs):
    B_r=inputs['B_r']
    mu_r=inputs['mu_r']
    g_eq=inputs['g_eq']
    t_mag=inputs['t_mag']

    outputs['B_g'] = B_r/(1+mu_r*g_eq/t_mag)    # neglecting leakage flux and fringing, magnetic voltag drop in steel (eqn2.14 Gieres PMSM) - g should be effective gap, not mechanical g