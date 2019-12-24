from __future__ import absolute_import
import numpy as np
from math import pi
from openmdao.api import Problem, IndepVarComp, ExplicitComponent, ExecComp
from openmdao.api import NewtonSolver, Group, DirectSolver, NonlinearRunOnce, LinearRunOnce, view_model, BalanceComp, ScipyOptimizeDriver



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

        outputs['P_in'] = I*V*np.sqrt(2) # Tq * omega * total_losses
        outputs['P_out'] = Tq*omega
