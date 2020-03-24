# Winding coeff in the specific electrical loading artifically low to match motor-cad report


from __future__ import absolute_import
import numpy as np
from math import pi

import openmdao.api as om

class TorqueComp(om.ExplicitComponent):

    def setup(self):
       self.add_input('B_g', 1, units='T', desc='air gap flux density')    
       self.add_input('n_m', 20, desc='number of magnets')
       self.add_input('n_turns', 12, desc='number of wire turns')
       self.add_input('I', 35, units='A', desc='RMS current')       
       self.add_input('rot_or', 0.060, units='m', desc='rotor outer radius')
       self.add_input('sta_ir', 0.060, units='m', desc='stator inner radius')
       self.add_input('P_shaft', 14000, units='W', desc='output power') 
       self.add_input('rpm', 5000, units='rpm', desc='Rotational Speed')
       self.add_input('stack_length', .0345, units='m', desc='stack length')

       self.add_output('Tq_shaft', 25, units='N*m', desc='torque')
       self.add_output('rot_volume', 0.2, units='m**3', desc='rotor volume')
       self.add_output('stator_surface_current', 51860, units='A/m', desc='specific electrical loading')
       self.add_output('omega', 1000, units='Hz', desc='mechanical rad/s')
       self.add_output('Tq_max', 30, units='N*m', desc='max torque available')
       
       #self.declare_partials('Tq', ['n_m','n','B_g','stack_length','rot_or','i'])

       self.declare_partials('*','*', method='fd')

    def compute(self,inputs,outputs):
       n_m=inputs['n_m']
       n_turns= inputs['n_turns']
       B_g= inputs['B_g']
       rot_or = inputs['rot_or']
       I = inputs['I']
       sta_ir = inputs['sta_ir']
       rpm = inputs['rpm']
       P_shaft = inputs['P_shaft']
       stack_length = inputs['stack_length']

       outputs['omega'] = rpm*2*pi/60
       outputs['stator_surface_current'] = 6 * 0.75*96/(2*sta_ir*np.pi) * I*np.sqrt(2)    # 0.75 represents the winding factor. This low value is required to match SEL from motor-cad
       outputs['Tq_shaft'] = P_shaft/outputs['omega']
       outputs['Tq_max'] = stack_length*2*n_m*n_turns*B_g*rot_or*I    # Eqn 4.11, pg 79, from D.Hansleman book
       outputs['rot_volume'] = (np.pi * rot_or**2 * stack_length)

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


class EfficiencyComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('P_wire', 500, units='W', desc='copper losses')
        self.add_input('P_steinmetz', 500, units='W', desc='iron losses')  
        self.add_input('P_shaft', 14000, units='W', desc='output power') 
        self.add_input('Tq_shaft', 30, units='N*m', desc='torque') 
        self.add_input('omega', 1000, units='Hz', desc='mechanical rad/s')  
        self.add_input('rpm', 5000, units='rpm', desc='speed of prop')   

        self.add_output('P_in', 15, units='kW', desc='input power')
        self.add_output('Eff', 0.90, desc='efficiency of motor')
        
        self.declare_partials('*','*', method='fd')


    def compute(self, inputs, outputs):
        rpm = inputs['rpm']
        P_wire = inputs['P_wire']
        P_steinmetz =inputs['P_steinmetz']
        P_shaft = inputs['P_shaft']
        Tq_shaft = inputs['Tq_shaft']
        omega = inputs['omega']

        outputs['P_in']  = (Tq_shaft*omega) + P_wire + P_steinmetz       

        outputs['Eff'] =  P_shaft / outputs['P_in']