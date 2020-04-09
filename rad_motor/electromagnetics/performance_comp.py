# Winding coeff in the specific electrical loading artifically low to match motor-cad report


from __future__ import absolute_import
import numpy as np
from math import pi

import openmdao.api as om

class TorqueComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
       nn = self.options['num_nodes']
       self.add_input('B_g', 1, units='T', desc='air gap flux density')    
       self.add_input('n_m', 20, desc='number of magnets')
       self.add_input('n_turns', 11, desc='number of wire turns')      
       self.add_input('rot_or', 0.060, units='m', desc='rotor outer radius')
       self.add_input('P_shaft', 12000*np.ones(nn), units='W', desc='output power') 
       self.add_input('rpm', 1000*np.ones(nn), units='rpm', desc='Rotational Speed')
       self.add_input('stack_length', .0345, units='m', desc='stack length')

       self.add_output('omega', 900*np.ones(nn), units='Hz', desc='mechanical rad/s')     
       self.add_output('Tq_shaft', 25*np.ones(nn), units='N*m', desc='torque')
       self.add_output('I_required', 30*np.ones(nn), units='A', desc='max torque available')      
       
       r = c = np.arange(nn)  # for scalar variables only
       self.declare_partials('omega', 'rpm', rows=r, cols=c)
       self.declare_partials('Tq_shaft', ['P_shaft', 'rpm'], rows=r, cols=c)
       self.declare_partials('I_required', ['stack_length', 'n_m', 'n_turns', 'B_g', 'rot_or'])
       self.declare_partials('I_required', ['P_shaft', 'rpm'], rows=r, cols=c)

    def compute(self,inputs,outputs):
        n_m=inputs['n_m']
        n_turns= inputs['n_turns']
        B_g= inputs['B_g']
        rot_or = inputs['rot_or']
        rpm = inputs['rpm']
        P_shaft = inputs['P_shaft']
        stack_length = inputs['stack_length'] 

        outputs['omega'] = rpm*2*pi/60 
        outputs['Tq_shaft'] = P_shaft/(rpm*2*pi/60)
        # outputs['Tq_max'] = stack_length*2*n_m*n_turns*B_g*rot_or*I    # Eqn 4.11, pg 79, from D.Hansleman book
        outputs['I_required'] = stack_length*2*n_m*n_turns*B_g*rot_or*outputs['Tq_shaft'] + 3

    def compute_partials(self,inputs,J):
        n_m=inputs['n_m']
        n_turns= inputs['n_turns']
        B_g= inputs['B_g']
        rot_or = inputs['rot_or']
        rpm = inputs['rpm']
        P_shaft = inputs['P_shaft']
        stack_length = inputs['stack_length']

        J['omega', 'rpm'] = 2*pi/60 

        J['Tq_shaft', 'P_shaft'] = 1/(rpm*2*pi/60)
        J['Tq_shaft', 'rpm'] = -P_shaft/(rpm**2*2*pi/60)

        J['I_required', 'stack_length'] = 2*n_m*n_turns*B_g*rot_or*P_shaft/(rpm*2*pi/60)
        J['I_required', 'n_m'] = stack_length*2*n_turns*B_g*rot_or*P_shaft/(rpm*2*pi/60)
        J['I_required', 'n_turns'] = stack_length*2*n_m*B_g*rot_or*P_shaft/(rpm*2*pi/60)
        J['I_required', 'B_g'] = stack_length*2*n_m*n_turns*rot_or*P_shaft/(rpm*2*pi/60)
        J['I_required', 'rot_or'] = stack_length*2*n_m*n_turns*B_g*P_shaft/(rpm*2*pi/60)
        J['I_required', 'P_shaft'] = stack_length*2*n_m*n_turns*B_g*rot_or/(rpm*2*pi/60)
        J['I_required', 'rpm'] = -stack_length*2*n_m*n_turns*B_g*rot_or*P_shaft/(rpm**2*2*pi/60)

class EfficiencyComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('P_wire', 500*np.ones(nn), units='W', desc='copper losses')
        self.add_input('P_steinmetz', 200*np.ones(nn), units='W', desc='iron losses')  
        self.add_input('P_shaft', 14000*np.ones(nn), units='W', desc='output power') 
        self.add_input('Tq_shaft', 30*np.ones(nn), units='N*m', desc='torque') 
        self.add_input('omega', 400*np.ones(nn), units='Hz', desc='mechanical rad/s')  
        self.add_input('rpm', 5000*np.ones(nn), units='rpm', desc='speed of prop')   

        self.add_output('P_in', 10000*np.ones(nn), units='W', desc='input power')
        self.add_output('Eff', 0.90*np.ones(nn), desc='efficiency of motor')
        
        r = c = np.arange(nn)
        self.declare_partials('P_in', ['Tq_shaft', 'omega', 'P_wire', 'P_steinmetz'], rows=r, cols=c)
        self.declare_partials('Eff', ['P_shaft', 'Tq_shaft', 'omega', 'P_wire', 'P_steinmetz'], rows=r, cols=c)

    def compute(self, inputs, outputs):
        rpm = inputs['rpm']
        P_wire = inputs['P_wire']
        P_steinmetz =inputs['P_steinmetz']
        P_shaft = inputs['P_shaft']
        Tq_shaft = inputs['Tq_shaft']
        omega = inputs['omega']

        outputs['P_in']  = (Tq_shaft*omega) + P_wire + (P_steinmetz)       
        outputs['Eff'] =  P_shaft / outputs['P_in']

    def compute_partials(self, inputs, J):
        rpm = inputs['rpm']
        P_wire = inputs['P_wire']
        P_steinmetz =inputs['P_steinmetz']
        P_shaft = inputs['P_shaft']
        Tq_shaft = inputs['Tq_shaft']
        omega = inputs['omega']

        J['P_in', 'Tq_shaft'] = omega  
        J['P_in', 'omega'] = Tq_shaft
        J['P_in', 'P_wire'] = 1
        J['P_in', 'P_steinmetz'] = 1

        J['Eff', 'P_shaft'] = 1 / ( (Tq_shaft*omega) + P_wire + P_steinmetz  )
        J['Eff', 'Tq_shaft'] = -omega*P_shaft / ( (Tq_shaft*omega) + P_wire + P_steinmetz  )**2
        J['Eff', 'omega'] = -Tq_shaft*P_shaft / ( (Tq_shaft*omega) + P_wire + P_steinmetz  )**2
        J['Eff', 'P_wire'] = -P_shaft / ( (Tq_shaft*omega) + P_wire + P_steinmetz  )**2
        J['Eff', 'P_steinmetz'] = -P_shaft / ( (Tq_shaft*omega) + P_wire + (P_steinmetz) )**2