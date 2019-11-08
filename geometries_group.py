from __future__ import absolute_import
import numpy as np
from math import pi
from openmdao.api import Problem, IndepVarComp, ExplicitComponent, ExecComp
from openmdao.api import NewtonSolver, Group, DirectSolver, NonlinearRunOnce, LinearRunOnce, view_model, BalanceComp
from efficiency_group import EfficiencyGroup


class motor_size(ExplicitComponent):

    def setup(self):
        self.add_input('radius_motor', 0.0765, units='m', desc='outer radius of motor')
        self.add_input('gap', 0.001, units='m', desc='air gap')
        self.add_input('rot_or', .05, units='m', desc='rotor outer radius')
        self.add_input('b_g', 1.0, units='T', desc='air gap flux density')
        self.add_input('k', 0.95, desc='stacking factor')
        self.add_input('b_ry', 2.4, units='T', desc='flux density of stator yoke')
        self.add_input('n_m', 16, desc='number of poles')
        self.add_input('t_mag', 0.005, desc='magnet thickness')
        self.add_input('b_sy', 2.4, units='T', desc='flux density of stator yoke')
        self.add_input('b_t', 2.4, units='T', desc='flux density of tooth')
        self.add_input('n_s', 15, desc='Number of slots')
        self.add_input('n', 16, desc='number of wire turns')
        self.add_input('i', 30, units='A', desc='RMS current')
        self.add_input('k_wb', 0.65, desc='bare wire slot fill factor')

        self.add_output('J', units='A/mm**2', desc='Current density')
        self.add_output('w_ry', 1.0, units='m', desc='width of stator yoke')
        self.add_output('w_sy', .005, units='m', desc='width of stator yoke')
        self.add_output('w_t', 0.010, units='m', desc='width of tooth')
        self.add_output('sta_ir', units='m', desc='stator inner radius')
        self.add_output('rot_ir', units='m', desc='rotor inner radius')
        self.add_output('s_d', units='m', desc='slot depth')

        self.declare_partials('*','*', method='fd')
        #self.declare_partials('w_t', ['rot_or','b_g','n_s','k','b_t'])
        #self.declare_partials('w_sy', ['rot_or', 'b_g', 'n_m', 'k', 'b_sy'])
        #self.declare_partials('w_ry', ['rot_or', 'b_g', 'n_m', 'k', 'b_ry'])

    def compute(self,inputs,outputs):
        rot_or = inputs['rot_or']
        radius_motor = inputs['radius_motor']  # .0765
        gap = inputs['gap']
        b_g = inputs['b_g']
        n_m = inputs['n_m']
        k = inputs['k']
        b_ry = inputs['b_ry']
        t_mag = inputs['t_mag']
        n = inputs['n']
        i = inputs['i']
        k_wb = inputs['k_wb']
        b_sy= inputs['b_sy']
        n_s = inputs['n_s']
        b_t = inputs['b_t']

        outputs['w_ry'] = (pi*rot_or*b_g)/(n_m*k*b_ry) 
        outputs['w_t'] = (2*pi*rot_or*b_g) / (n_s*k*b_t) 
        outputs['w_sy'] = (pi*rot_or*b_g)/(n_m*k*b_sy)
        outputs['s_d'] = radius_motor - rot_or - gap - outputs['w_sy']
        #outputs['radius_motor'] = rot_or + gap + s_d + outputs['w_sy']
        outputs['rot_ir'] = (rot_or- t_mag) - outputs['w_ry'] 
        outputs['sta_ir'] = rot_or + gap
        area = pi*(radius_motor-outputs['w_sy'])**2 - pi*(radius_motor-outputs['w_sy']-outputs['s_d'])**2
        outputs['J'] = 2*n*i*(2.**0.5)/(k_wb/n_s*(area-n_s*1.25*(outputs['w_t']*outputs['s_d']))*1E6)


    # def compute_partials(self, inputs, J):

    #     # rotor_outer_radius
    #     # radius_motor = inputs['radius_motor']
    #     # s_d = inputs['s_d']
    #     # w_sy = inputs['w_sy']
    #     # J['rot_or', 'radius_motor'] = 1-w_sy-s_d-gap
    #     # J['rot_or', 's_d'] = radius_motor - w_sy - 1 - gap
    #     # J['rot_or', 'w_sy'] = radius_motor - 1 - s_d - gap

    #     # rotor_yoke_width
    #     rot_or = inputs['rot_or']
    #     b_g= inputs['b_g']
    #     n_m= inputs['n_m']
    #     k = inputs['k']
    #     b_ry= inputs['b_ry']
    #     J['w_ry', 'rot_or'] = (pi*b_g)/(n_m*k*b_ry)
    #     J['w_ry', 'b_g'] = (pi*rot_or)/(n_m*k*b_ry)
    #     J['w_ry', 'n_m'] = -(pi*rot_or*b_g)/(n_m**3*k*b_ry)
    #     J['w_ry', 'k']   = -(pi*rot_or*b_g)/(n_m*k**2*b_ry)
    #     J['w_ry', 'b_ry'] = -(pi*rot_or*b_g)/(n_m*k*b_ry**2)

    #     # stator_yoke_width
    #     b_sy= inputs['b_sy']
    #     J['w_sy', 'rot_or'] = (pi*b_g)/(n_m*k*b_sy)
    #     J['w_sy', 'b_g'] = (pi*rot_or)/(n_m*k*b_sy)
    #     J['w_sy', 'n_m'] = -(pi*rot_or*b_g)/(n_m**3*k*b_sy)
    #     J['w_sy', 'k']   = -(pi*rot_or*b_g)/(n_m*k**2*b_sy)
    #     J['w_sy', 'b_sy'] = -(pi*rot_or*b_g)/(n_m*k*b_sy**2)

    #     # tooth_width
    #     n_s = inputs['n_s']
    #     b_t = inputs['b_t']
    #     J['w_t', 'rot_or'] = (2*pi*b_g)/(n_s*k*b_t)
    #     J['w_t', 'b_g'] = (2*pi*rot_or)/(n_s*k*b_t)
    #     J['w_t', 'n_s'] = -(2*pi*rot_or*b_g)/(n_s**2*k*b_t)
    #     J['w_t', 'k']   = -(2*pi*rot_or*b_g)/(n_s*k**2*b_t)
    #     J['w_t', 'b_t'] = -(2*pi*rot_or*b_g)/(n_s*k*b_t**2)


class motor_mass(ExplicitComponent):

    def setup(self):
        self.add_input('rho', 8110.2, units='kg/m**3', desc='density of hiperco-50')
        self.add_input('radius_motor', .075, units='m', desc='motor outer radius')           
        self.add_input('n_s', 15, desc='number of slots')                           
        self.add_input('sta_ir', .050, units='m', desc='stator inner radius')       
        self.add_input('w_t', units='m', desc='tooth width')                        
        self.add_input('stack_length', units='m', desc='length of stack')  
        self.add_input('s_d', units='m', desc='slot depth')                         
        self.add_input('rot_or', 0.0615, units='m', desc='rotor outer radius')
        self.add_input('rot_ir', 0.0515, units='m', desc='rotor inner radius')
        self.add_input('t_mag', .005, units='m', desc='magnet thickness')
        self.add_input('rho_mag', 7500, units='kg/m**3', desc='density of magnet')

        self.add_output('mag_mass', 0.5, units='kg', desc='mass of magnets')
        self.add_output('sta_mass', 25, units='kg', desc='mass of stator')
        self.add_output('rot_mass', 1.0, units='kg', desc='weight of rotor')
        
        self.declare_partials('*','*', method='fd')

    def compute(self,inputs,outputs):
        rho=inputs['rho']
        radius_motor=inputs['radius_motor']
        n_s=inputs['n_s']
        sta_ir=inputs['sta_ir']
        w_t=inputs['w_t']
        stack_length=inputs['stack_length']
        s_d=inputs['s_d']
        rot_ir=inputs['rot_ir']
        rot_or=inputs['rot_or']
        stack_length=inputs['stack_length']
        t_mag=inputs['t_mag']
        rho_mag=inputs['rho_mag']

        outputs['sta_mass'] = rho * stack_length * ((pi * radius_motor**2)-(pi * (sta_ir+s_d)**2)+(n_s*(w_t*s_d*1.5)))
        outputs['rot_mass'] = (pi*(rot_or - t_mag)**2 - pi*rot_ir**2) * rho * stack_length
        outputs['mag_mass'] = (((pi*rot_or**2) - (pi*(rot_or-t_mag)**2))) * rho_mag * stack_length

    # def compute_partials(self,inputs,J):

        # stator
    #   rho=inputs['rho']
    #   radius_motor=inputs['radius_motor']
    #   n_s=inputs['n_s']
    #   sta_ir=inputs['sta_ir']
    #   w_t=inputs['w_t']
    #   stack_length=inputs['stack_length']

    #   J['sta_mass', 'rho'] = 
    #   J['sta_mass', 'radius_motor'] = 
    #   J['sta_mass', 'n_s'] = 
    #   J['sta_mass', 'sta_ir'] = 
    #   J['sta_mass', 'w_t'] = 
    #   J['sta_mass', 'stack_length'] = 

# ---------------------------------------        



