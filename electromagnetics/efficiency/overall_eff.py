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
