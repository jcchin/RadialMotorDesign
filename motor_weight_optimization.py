from __future__ import absolute_import
import numpy as np
import pandas as pd
import math
from openmdao.api import Problem, IndepVarComp, ExplicitComponent, ExecComp
from openmdao.api import NewtonSolver, Group, DirectSolver, NonlinearRunOnce, LinearRunOnce, view_model
from openmdao.components.meta_model_structured_comp import MetaModelStructuredComp

class rotor_outer_radius(ExplicitComponent):

	def setup(self):
		self.add_input('r_m', 0.25, units='m', desc='outer radius of motor')
		self.add_input('s_d', 0.015, units='m', desc='slot depth')
		self.add_input('w_sy', 0.005, units='m', desc='stator yoke width')
		self.add_output('rot_or', .05, units='m', desc='rotor outer radius')
		self.declare_partials('rot_or', ['r_m','s_d','w_sy','w_ry'])

	def compute(self,inputs,oututs):
		r_m = inputs['r_m']
		s_d = inputs['s_d']
		w_sy = inputs['w_sy']

		outputs['rot_or'] = r_m - w_sy - s_d - .001

	def compute_partials(self, inputs,J):
		r_m = inputs['r_m']
		s_d = inputs['s_d']
		w_sy = inputs['w_sy']

		J['rot_or', 'r_m'] = 1-w_sy-s_d-.001
		J['rot_or', 's_d'] = r_m - w_sy - 1 - .001
		J['rot_or', 'w_sy'] = r_m - 1 - s_d - .001


class rotor_yoke_width(ExplicitComponent):

	def setup(self):
		self.add_input('rot_or', .05, units='m', desc='rotor outer radius')
		self.add_input('b_g', 1.0, units='T', desc='air gap flux density')
		self.add_input('k', 0.95, desc='stacking factor')
		self.add_input('b_ry', 2.4, units='T', desc='flux density of stator yoke')
		self.add_input('n_m', 16, desc='number of poles')
		self.add_output('w_ry', 1.0, units='m', desc='width of stator yoke')
		self.declare_partials('w_ry', ['rot_or', 'b_g', 'n_m', 'k', 'b_ry'])


	def compute(self,inputs,outputs):
		rot_or = inputs['rot_or']
		b_g= inputs['b_g']
		n_m= inputs['n_m']
		k = inputs['k']
		b_ry= inputs['b_ry']

		outputs['w_ry'] = (math.pi*rot_or*b_g)/(n_m*k*b_ry) 

	def compute_partials(self,inputs,J):
		rot_or = inputs['rot_or']
		b_g= inputs['b_g']
		n_m= inputs['n_m']
		k = inputs['k']
		b_ry= inputs['b_ry']

		J['w_ry', 'rot_or'] = (math.pi*b_g)/(n_m*k*b_ry)
		J['w_ry', 'b_g'] = (math.pi*rot_or)/(n_m*k*b_ry)
		J['w_ry', 'n_m'] = -(math.pi*rot_or*b_g)/(n_m**3*k*b_ry)
		J['w_ry', 'k']   = -(math.pi*rot_or*b_g)/(n_m*k**2*b_ry)
		J['w_ry', 'b_ry'] = -(math.pi*rot_or*b_g)/(n_m*k*b_ry**2)


class stator_yoke_width(ExplicitComponent):

	def setup(self):
		self.add_input('b_sy', 2.4, units='T', desc='flux density of stator yoke')
		self.add_input('k', 0.95, desc='stacking factor')
		self.add_input('b_g', 1, units='T', desc='flux density of air gap')
		self.add_input('n_m', 16, desc='number of poles')		
		self.add_input('rot_or', 0.025, units='m', desc='rotor outer radius')
		self.add_output('w_sy', .005, units='m', desc='width of stator yoke')
		self.declare_partials('w_sy', ['rot_or', 'b_g', 'n_m', 'k', 'b_sy'])

	def compute(self,inputs,outputs):
		rot_or = inputs['rot_or']
		b_g= inputs['b_g']
		n_m= inputs['n_m']
		k = inputs['k']
		b_sy= inputs['b_sy']
		outputs['w_sy'] = (math.pi*rot_or*b_g)/(n_m*k*b_sy)

	def compute_partials(self,inputs,J):
		rot_or = inputs['rot_or']
		b_g= inputs['b_g']
		n_m= inputs['n_m']
		k= inputs['k']
		b_sy= inputs['b_sy']

		J['w_sy', 'rot_or'] = (math.pi*b_g)/(n_m*k*b_sy)
		J['w_sy', 'b_g'] = (math.pi*rot_or)/(n_m*k*b_sy)
		J['w_sy', 'n_m'] = -(math.pi*rot_or*b_g)/(n_m**3*k*b_sy)
		J['w_sy', 'k']   = -(math.pi*rot_or*b_g)/(n_m*k**2*b_sy)
		J['w_sy', 'b_sy'] = -(math.pi*rot_or*b_g)/(n_m*k*b_sy**2)


class tooth_width(ExplicitComponent):

	def setup(self):
		self.add_input('b_t', 2.4, units='T', desc='flux density of tooth')
		self.add_input('k', 0.95, desc='stacking factor')
		self.add_input('b_g', 1, units='T', desc='air gap flux density')
		self.add_input('n_s', 15, desc='Number of slots')
		self.add_input('rot_or', .025, units='m', desc='outer radius of rotor')
		self.add_output('w_t', 0.010, units='m', desc='width of tooth')
		self.declare_partials('w_t', ['rot_or','b_g','n_s','k','b_t'])		

	def compute(self,inputs,outputs):
		rot_or = inputs['rot_or']
		b_g = inputs['b_g']
		n_s = inputs['n_s']
		k = inputs['k']
		b_t = inputs['b_t']
		outputs['w_t'] = (2*math.pi*rot_or*b_g) / (n_s*k*b_t)

	def compute_partials(self,inputs,J):

		rot_or = inputs['rot_or']
		b_g = inputs['b_g']
		n_s = inputs['n_s']
		k = inputs['k']
		b_t = inputs['b_t']
		
		J['w_t', 'rot_or'] = (2*math.pi*b_g)/(n_s*k*b_t)
		J['w_t', 'b_g'] = (2*math.pi*rot_or)/(n_s*k*b_t)
		J['w_t', 'n_s'] = -(2*math.pi*rot_or*b_g)/(n_s**2*k*b_t)
		J['w_t', 'k']   = -(2*math.pi*rot_or*b_g)/(n_s*k**2*b_t)
		J['w_t', 'b_t'] = -(2*math.pi*rot_or*b_g)/(n_s*k*b_t**2)


class torque(ExplicitComponent):

	def setup(self):
		self.add_input('b_g', 2.4, units='T', desc='air gap flux density')	
		self.add_input('n_m', 16, desc='number of poles')
		self.add_input('n', 16, desc='number of wire turns')
		self.add_input('l_st', .045, units='m', desc='stack length')
		self.add_input('i', 30, units='A', desc='RMS current')		
		self.add_input('rot_or', .025, units='m', desc='rotor outer radius')
		self.add_output('tq', 25, units='N*m', desc='torque')
		self.declare_partials('tq', ['n_m','n','b_g','l_st','rot_or','i'])

	def compute(self,inputs,outputs):
		n_m=inputs['n_m']
		n= inputs['n']
		b_g= inputs['b_g']
		l_st= inputs['l_st']
		rot_or = inputs['rot_or']
		i = inputs['i']

		outputs['tq'] = 2*n_m*n*b_g*l_st*rot_or*i*.68

	def compute_partials(self,inputs,J):
		n_m=inputs['n_m']
		n= inputs['n']
		b_g= inputs['b_g']
		l_st= inputs['l_st']
		rot_or = inputs['rot_or']
		i = inputs['i']

		J['tq','n_m'] = 2*n*b_g*l_st*rot_or*i
		J['tq', 'n'] = 2*n_m*b_g*l_st*rot_or*i
		J['tq', 'b_g'] = 2*n_m*n*l_st*rot_or*i
		J['tq', 'l_st'] = 2*n_m*n*b_g*rot_or*i
		J['tq', 'rot_or'] = 2*n_m*n*b_g*l_st*i
		J['tq', 'i'] = 2*n_m*n*b_g*l_st*rot_or


class stator_mass(ExplicitComponent):

	def setup(self):
		self.add_input('rho', 8110.2, units='kg/m**3', desc='density of hiperco-50')
		self.add_input('r_m', .075, units='m', desc='motor outer radius')           
		self.add_input('n_s', 15, desc='number of slots')                           
		self.add_input('sta_ir', .050, units='m', desc='stator inner radius')       
		self.add_input('w_t', units='m', desc='tooth width')                        
		self.add_input('l_st', units='m', desc='length of stack')  
		self.add_input('s_d', units='m', desc='slot depth')                 
		self.add_output('sta_mass', 25, units='kg', desc='mass of stator')
		self.declare_partials('sta_mass', ['rho','r_m','n_s','sta_ir','w_t','l_st','s_d'])

	def compute(self,inputs,outputs):
		rho=inputs['rho']
		r_m=inputs['r_m']
		n_s=inputs['n_s']
		sta_ir=inputs['sta_ir']
		w_t=inputs['w_t']
		l_st=inputs['l_st']
		s_d=inputs['s_d']

		outputs['sta_mass'] = rho * l_st * ((math.pi * r_m**2)-(math.pi * (sta_ir+s_d)**2)+(n_s*(w_t*s_d*1.5)))

	# def compute_partials(self,inputs,J):
	# 	rho=inputs['rho']
	# 	r_m=inputs['r_m']
	# 	n_s=inputs['n_s']
	# 	sta_ir=inputs['sta_ir']
	# 	w_t=inputs['w_t']
	# 	l_st=inputs['l_st']

	# 	J['sta_mass', 'rho'] = 
	# 	J['sta_mass', 'r_m'] = 
	# 	J['sta_mass', 'n_s'] = 
	# 	J['sta_mass', 'sta_ir'] = 
	# 	J['sta_mass', 'w_t'] = 
	# 	J['sta_mass', 'l_st'] = 

class rotor_mass(ExplicitComponent):
	def setup(self):
		self.add_input('rho', 8110.2, units='kg/m**3', desc='density of hiperco-50')
		self.add_input('rot_or', 0.0615, units='m', desc='rotor outer radius')
		self.add_input('rot_ir', 0.0515, units='m', desc='rotor inner radius')
		self.add_input('l_st', 0.038, units='m', desc='stack length')
		self.add_output('rot_mass', 1.0, units='kg', desc='weight of rotor')
		# self.declare_partials('rot_mass',['rho','rot_or','rot_ir'])

	def compute(self,inputs,outputs):
		rho=inputs['rho']
		rot_ir=inputs['rot_ir']
		rot_or=inputs['rot_or']
		l_st=inputs['l_st']

		outputs['rot_mass'] = (math.pi*rot_or**2 - math.pi*rot_ir**2) * rho * l_st

	# def compute_partials(self,inputs,J):


p = Problem()
model = p.model

ind = model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])

ind.add_output('rot_or', val=0.0615, units='m')			# Outer radius of rotor, including 5mm magnet thickness
ind.add_output('k', val=0.97)							# Stacking factor
ind.add_output('n', val=16)								# Number of wire turns		
ind.add_output('i', val=30, units='A')					# RMS Current

ind.add_output('b_g', val = 1, units='T')				# Air gap flux Density    !! Flux values may represent 100% slot fill !!
ind.add_output('b_ry', val=4, units='T')				# Rotor yoke flux density
ind.add_output('b_sy', val=4, units='T')				# Stator yoke flux density
ind.add_output('b_t', val=4, units='T')				# Tooth Flux Density

ind.add_output('n_s', val=15)							# Number of Slots
ind.add_output('n_m', val=16)							# Number of poles

ind.add_output('l_st', val=0.038, units='m')			# Stack Length
ind.add_output('rho', val=8110.2, units='kg/m**3')		# Density of Hiperco-50


model.add_subsystem('rotor_yw', rotor_yoke_width(), promotes_inputs=['rot_or','b_g','k','b_ry','n_m'], promotes_outputs=['w_ry'])
model.add_subsystem('stator_yoke_width', stator_yoke_width(), promotes_inputs=['rot_or','b_g','k','b_sy','n_m'], promotes_outputs=['w_sy'])
model.add_subsystem('slot_depth', ExecComp('s_d = .0765-rot_or-0.001 - w_sy',w_sy={'units':'m'}, s_d={'units':'m'},rot_or={'units':'m'}), promotes_inputs=['rot_or','w_sy'], promotes_outputs=['s_d'])
model.add_subsystem('motor_radius', ExecComp('r_m = rot_or + .001 + s_d + w_sy',w_sy={'units':'m'},r_m={'units':'m'}, rot_or={'units':'m'}, s_d={'units':'m'}), promotes_inputs=['rot_or','s_d','w_sy'], promotes_outputs=['r_m'])
model.add_subsystem('rotor_in_radius', ExecComp('rot_ir = rot_or - w_ry - .005',rot_or={'units':'m'}, rot_ir={'units':'m'}, w_ry={'units':'m'}), promotes_inputs=['rot_or', 'w_ry'], promotes_outputs=['rot_ir'])
model.add_subsystem('stator_in_radius', ExecComp('sta_ir = rot_or + .001', rot_or={'units':'m'},sta_ir={'units':'m'}), promotes_inputs=['rot_or'], promotes_outputs=['sta_ir'])
model.add_subsystem('tooth_width', tooth_width(), promotes_inputs=['rot_or','b_t','k','n_s','b_g'], promotes_outputs=['w_t'])
model.add_subsystem('torque', torque(), promotes_inputs=['rot_or','b_g', 'i','n_m','n','l_st'], promotes_outputs=['tq'])
model.add_subsystem('stator_mass', stator_mass(), promotes_inputs=['rho','r_m','n_s','sta_ir','w_t','l_st','s_d'], promotes_outputs=['sta_mass'])
model.add_subsystem('rotor_mass', rotor_mass(), promotes_inputs=['rho','rot_or','rot_ir','l_st'], promotes_outputs=['rot_mass'])


# model.add_subsystem('motor_radius_prime', ExecComp('r_m_p = rot_or + .005 + .001 + s_d + w_sy',r_m_p={'units':'m'}, rot_or={'units':'m'}, s_d={'units':'m'}, w_sy={'units':'m'}), promotes_inputs=['rot_or','s_d','w_sy'], promotes_outputs=['r_m_p'])


# model.add_subsystem('mass_stator', mass_stator(), promotes_inputs=['rho','r_m','n_s','sta_ir','w_t','l_st'], promotes_outputs=['weight']
# model.add_subsystem('stmass', ExecComp('mass = l_st * ((math.pi * r_m**2)-(math.pi * sta_ir**2)+(n_s*(w_t*1.2)))', l_st={'units':'m'},r_m={'units':'m'},sta_ir={'units':'m'},w_t={'units':'m'}), promotes_inputs=['l_st','r_m','sta_ir','n_s','w_t'], promotes_outputs=['mass']

p.setup()
p.run_model()

print('Rotor Inner Radius................',  p.get_val('rot_ir', units='mm'))

print('Stator Inner Radius...............',  p.get_val('sta_ir', units='mm'))
print('Motor Outer Radius................',  p.get_val('motor_radius.r_m', units='mm'))

print('Rotor Yoke Thickness..............',  p.get_val('w_ry', units='mm'))
print('Slot Depth........................',  p.get_val('s_d', units='mm'))
print('Stator Yoke Thickness.............',  p.get_val('w_ry', units='mm'))
print('Tooth Width.......................',  p.get_val('w_t', units='mm'))

print('Torque............................',  p['tq'])

print('Mass of Stator....................',  p.get_val('sta_mass', units='kg'))
print('Mass of Rotor.....................',  p.get_val('rot_mass', units='kg'))

