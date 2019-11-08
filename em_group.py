from __future__ import absolute_import
import numpy as np
from math import pi
from openmdao.api import Problem, IndepVarComp, ExplicitComponent, ExecComp
from openmdao.api import NewtonSolver, Group, DirectSolver, NonlinearRunOnce, LinearRunOnce, view_model, BalanceComp, ScipyOptimizeDriver



class EfficiencyGroup(ExplicitComponent):
	def setup(self):
		self.add_input('i', 30, units='A', desc='RMS current')
		self.add_input('tq', 25, units='N*m', desc='torque')
		self.add_input('v', 385, units='V', desc='RMS voltage')

		self.add_output('P_in', 15, units='kW', desc='input power')
		self.add_output('P_out', 15, units='kW', desc='output power')
		
		self.declare_partials('*','*', method='fd')


	def compute(self, inputs, outputs):
		i = inputs['i']
		tq = inputs['tq']
		v = inputs['v']

		omega = rm*(2*pi/60)

		outputs['P_in'] = (3**.5)*v*i
		outputs['P_out'] = tq*omega




class torque(ExplicitComponent):

    def setup(self):
       self.add_input('b_g', 2.4, units='T', desc='air gap flux density')    
       self.add_input('n_m', 16, desc='number of magnets')
       self.add_input('n', 16, desc='number of wire turns')
       self.add_input('stack_length', .0345, units='m', desc='stack length')
       self.add_input('i', 30, units='A', desc='RMS current')       
       self.add_input('rot_or', .025, units='m', desc='rotor outer radius')
       self.add_output('tq', 25, units='N*m', desc='torque')
       #self.declare_partials('tq', ['n_m','n','b_g','stack_length','rot_or','i'])
       self.declare_partials('*','*', method='fd')

    def compute(self,inputs,outputs):
       n_m=inputs['n_m']
       n= inputs['n']
       b_g= inputs['b_g']
       stack_length= inputs['stack_length']
       rot_or = inputs['rot_or']
       i = inputs['i']

       outputs['tq'] = 2*n_m*n*b_g*stack_length*rot_or*i*.68

    # def compute_partials(self,inputs,J):
    #    n_m=inputs['n_m']
    #    n= inputs['n']
    #    b_g= inputs['b_g']
    #    stack_length= inputs['stack_length']
    #    rot_or = inputs['rot_or']
    #    i = inputs['i']

    #    J['tq','n_m'] = 2*n*b_g*stack_length*rot_or*i
    #    J['tq', 'n'] = 2*n_m*b_g*stack_length*rot_or*i
    #    J['tq', 'b_g'] = 2*n_m*n*stack_length*rot_or*i
    #    J['tq', 'stack_length'] = 2*n_m*n*b_g*rot_or*i
    #    J['tq', 'rot_or'] = 2*n_m*n*b_g*stack_length*i
    #    J['tq', 'i'] = 2*n_m*n*b_g*stack_length*rot_or