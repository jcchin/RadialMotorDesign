# Winding coeff in the specific electrical loading artifically low to match motor-cad report


from __future__ import absolute_import
import numpy as np
from math import pi
from openmdao.api import Problem, IndepVarComp, ExplicitComponent, ExecComp
from openmdao.api import NewtonSolver, Group, DirectSolver, NonlinearRunOnce, LinearRunOnce, view_model, BalanceComp, ScipyOptimizeDriver



class Efficiency(ExplicitComponent):
	def setup(self):
		self.add_input('i', 30, units='A', desc='RMS current')
		self.add_input('tq', 25, units='N*m', desc='torque')
		self.add_input('v', 385, units='V', desc='RMS voltage')
		self.add_input('rpm', 5000, units='rpm', desc='Rotational Speed')

		self.add_output('P_in', 15, units='kW', desc='input power')
		self.add_output('P_out', 15, units='kW', desc='output power')
		
		self.declare_partials('*','*', method='fd')


	def compute(self, inputs, outputs):
		i = inputs['i']
		tq = inputs['tq']
		v = inputs['v']
		rpm = inputs['rpm']

		omega = rpm*(2*pi/60)

		outputs['P_in'] = i*v*np.sqrt(2)
		outputs['P_out'] = tq*omega




class torque(ExplicitComponent):

    def setup(self):
       self.add_input('B_g', 2.4, units='T', desc='air gap flux density')    
       self.add_input('n_m', 16, desc='number of magnets')
       self.add_input('n_turns', 16, desc='number of wire turns')
       self.add_input('stack_length', .0345, units='m', desc='stack length')
       self.add_input('i', 30, units='A', desc='RMS current')       
       self.add_input('rot_or', .025, units='m', desc='rotor outer radius')
       self.add_input('sta_ir', 0.25, units='m', desc='stator inner radius')

       self.add_output('tq', 25, units='N*m', desc='torque')
       self.add_output('rot_volume', 0.2, units='m**3', desc='rotor volume')
       self.add_output('stator_surface_current', 51860, units='A/m', desc='specific electrical loading')
       #self.declare_partials('tq', ['n_m','n','B_g','stack_length','rot_or','i'])

       self.declare_partials('*','*', method='fd')

    def compute(self,inputs,outputs):
       n_m=inputs['n_m']
       n_turns= inputs['n_turns']
       B_g= inputs['B_g']
       stack_length= inputs['stack_length']
       rot_or = inputs['rot_or']
       i = inputs['i']
       sta_ir = inputs['sta_ir']

       outputs['rot_volume'] = (np.pi * sta_ir**2 * stack_length)
       outputs['stator_surface_current'] = 6 * 0.645*96/(2*sta_ir*np.pi) * i*np.sqrt(2)    # 0.75 represents the winding factor. This low value is required to match SEL from motor-cad

       outputs['tq'] = outputs['rot_volume'] * B_g* outputs['stator_surface_current'] * np.cos(0)   # Lipo, Pg. 372 # 6==constant; 0.933==winding factor; 96==turns per phase; 50==I peak; 1==cos(epsilon) when epsilon=0


    # def compute_partials(self,inputs,J):
    #    n_m=inputs['n_m']
    #    n= inputs['n']
    #    B_g= inputs['B_g']
    #    stack_length= inputs['stack_length']
    #    rot_or = inputs['rot_or']
    #    i = inputs['i']

    #    J['tq','n_m'] = 2*n*B_g*stack_length*rot_or*i
    #    J['tq', 'n'] = 2*n_m*B_g*stack_length*rot_or*i
    #    J['tq', 'B_g'] = 2*n_m*n*stack_length*rot_or*i
    #    J['tq', 'stack_length'] = 2*n_m*n*B_g*rot_or*i
    #    J['tq', 'rot_or'] = 2*n_m*n*B_g*stack_length*i
    #    J['tq', 'i'] = 2*n_m*n*B_g*stack_length*rot_or

    # class pmsm_torque(ExplicitComponent):

    #   def setup(self):
    #     self.add_input('n_phase', units=None, desc='number of phases')
    #     self.add_input('pole_pairs', units=None, desc='number of pole pairs')
    #     self.add_input('wind_fact', units='V', desc='winding factor')
    #     self.add_input('n_turns_phase', units=None, desc='number of wire turns per phase')

    #     self.add_output('torque_const', units=None, desc='torque constant')

    #     self.add_input('')

class field_intsities(ExplicitComponent):

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


class flux_densities(ExplicitComponent):

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