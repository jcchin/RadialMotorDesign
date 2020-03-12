# Winding coeff in the specific electrical loading artifically low to match motor-cad report


from __future__ import absolute_import
import numpy as np
from math import pi
from openmdao.api import Problem, IndepVarComp, ExplicitComponent, ExecComp
from openmdao.api import NewtonSolver, Group, DirectSolver, NonlinearRunOnce, LinearRunOnce, view_model, BalanceComp, ScipyOptimizeDriver

class TorqueComp(ExplicitComponent):

    def setup(self):
       self.add_input('B_g', 2.4, units='T', desc='air gap flux density')    
       self.add_input('n_m', 16, desc='number of magnets')
       self.add_input('n_turns', 16, desc='number of wire turns')
       self.add_input('stack_length', .0345, units='m', desc='stack length')
       self.add_input('I', 30, units='A', desc='RMS current')       
       self.add_input('rot_or', .025, units='m', desc='rotor outer radius')
       self.add_input('sta_ir', 0.25, units='m', desc='stator inner radius')

       self.add_output('Tq', 25, units='N*m', desc='torque')
       self.add_output('rot_volume', 0.2, units='m**3', desc='rotor volume')
       self.add_output('stator_surface_current', 51860, units='A/m', desc='specific electrical loading')
       #self.declare_partials('Tq', ['n_m','n','B_g','stack_length','rot_or','i'])

       self.declare_partials('*','*', method='fd')

    def compute(self,inputs,outputs):
       n_m=inputs['n_m']
       n_turns= inputs['n_turns']
       B_g= inputs['B_g']
       stack_length= inputs['stack_length']
       rot_or = inputs['rot_or']
       I = inputs['I']
       sta_ir = inputs['sta_ir']

       outputs['rot_volume'] = (np.pi * rot_or**2 * stack_length)
       outputs['stator_surface_current'] = 6 * 0.75*96/(2*sta_ir*np.pi) * I*np.sqrt(2)    # 0.75 represents the winding factor. This low value is required to match SEL from motor-cad

       outputs['Tq'] = outputs['rot_volume'] * B_g* outputs['stator_surface_current'] * np.cos(0)   # Lipo, Pg. 372 # 6==constant; 0.933==winding factor; 96==turns per phase; 50==I peak; 1==cos(epsilon) when epsilon=0
       # outputs['Tq'] = 10*3/2*I*.05

    # def compute_partials(self,inputs,J):
    #    n_m=inputs['n_m']
    #    n= inputs['n']
    #    B_g= inputs['B_g']
    #    stack_length= inputs['stack_length']
    #    rot_or = inputs['rot_or']
    #    i = inputs['i']

    #    J['Tq','n_m'] = 2*n*B_g*stack_length*rot_or*i
    #    J['Tq', 'n'] = 2*n_m*B_g*stack_length*rot_or*i
    #    J['Tq', 'B_g'] = 2*n_m*n*stack_length*rot_or*i
    #    J['Tq', 'stack_length'] = 2*n_m*n*B_g*rot_or*i
    #    J['Tq', 'rot_or'] = 2*n_m*n*B_g*stack_length*i
    #    J['Tq', 'i'] = 2*n_m*n*B_g*stack_length*rot_or

    # class pmsm_torque(ExplicitComponent):

    #   def setup(self):
    #     self.add_input('n_phase', units=None, desc='number of phases')
    #     self.add_input('pole_pairs', units=None, desc='number of pole pairs')
    #     self.add_input('wind_fact', units='V', desc='winding factor')
    #     self.add_input('n_turns_phase', units=None, desc='number of wire turns per phase')

    #     self.add_output('torque_const', units=None, desc='torque constant')

    #     self.add_input('')


class EfficiencyComp(ExplicitComponent):
    def setup(self):
        self.add_input('I', 30, units='A', desc='RMS current')
        self.add_input('Tq', 25, units='N*m', desc='torque') 
        self.add_input('V', 385, units='V', desc='RMS voltage')
        self.add_input('rpm', 5000, units='rpm', desc='Rotational Speed')

        self.add_output('P_in', 15, units='kW', desc='input power')
        self.add_output('P_out', 15, units='kW', desc='output power')
        
        self.declare_partials('*','*', method='fd')


    def compute(self, inputs, outputs):
        I = inputs['I']
        Tq = inputs['Tq']
        V = inputs['V']
        rpm = inputs['rpm']

        omega = rpm*(2*pi/60)  # mechanical rad/s

        outputs['P_in'] = V*I          #I*V*np.sqrt(2) # Tq * omega * total_losses
        outputs['P_out'] = Tq*omega
