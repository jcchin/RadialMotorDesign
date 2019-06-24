from __future__ import absolute_import
import numpy as np
from math import pi
from openmdao.api import Problem, IndepVarComp, ExplicitComponent, ExecComp
from openmdao.api import NewtonSolver, Group, DirectSolver, NonlinearRunOnce, LinearRunOnce, view_model, BalanceComp, ScipyOptimizeDriver

class efficiency(ExplicitComponent):

    def setup(self):
        self.add_input('i', 30, units='A', desc='RMS current')
        self.add_input('tq', 25, units='N*m', desc='torque')
        self.add_input('v', 500, units='V', desc='RMS voltage')
        self.add_input('rm', 5400, units='rpm', desc='motor speed')

        self.add_output('P_in', units='W', desc='input power')
        self.add_output('P_out', units='W', desc='output power')
        self.add_output('nu', desc='efficiency')
        
        self.declare_partials('*','*')

    def compute(self, inputs, outputs):
        i = inputs['i']
        tq = inputs['tq']
        v = inputs['v']
        rm = inputs['rm']
        
        outputs['P_in'] = (3**.5)*v*i
        outputs['P_out'] = tq*rm*(2*pi/60)
        outputs['nu'] = outputs['P_out']/outputs['P_in']
        
    def compute_partials(self, inputs, J):
        i = inputs['i']
        tq = inputs['tq']
        v = inputs['v']
        rm = inputs['rm']
        
        P_in = (3**.5)*v*i
        P_out = tq*rm*(2*pi/60)
        
        J['P_in','v'] = (3**.5)*i 
        J['P_in','i'] = (3**.5)*v 
        
        J['P_out','tq'] = rm*(2*pi/60)
        J['P_out','rm'] = tq*(2*pi/60)
        
        J['nu','v'] = -P_out/((3**.5)*i*(v**2))
        J['nu','i'] = -P_out/((3**.5)*v*(i**2))
        J['nu','tq'] = rm*(2*pi/60)/P_in
        J['nu','rm'] = tq*(2*pi/60)/P_in
        
        

class motor_size(ExplicitComponent):
        
    def setup(self):
        # rotor_outer_radius
        self.add_input('mot_or', 0.0765, units='m', desc='motor outer radius')
        self.add_input('gap', 0.001, units='m', desc='air gap')

        # rotor_yoke_width
        self.add_input('rot_or', .05, units='m', desc='rotor outer radius')
        self.add_input('b_g', 1.0, units='T', desc='air gap flux density')
        self.add_input('k', 0.95, desc='stacking factor')
        self.add_input('b_ry', 2.4, units='T', desc='flux density of stator yoke')
        self.add_input('n_m', 20, desc='number of poles')
        self.add_input('t_mag', 0.005, desc='magnet thickness')
        self.add_output('w_ry', 1.0, units='m', desc='width of rotor yoke')
        #self.declare_partials('w_ry', ['rot_or', 'b_g', 'n_m', 'k', 'b_ry'])

        # stator_yoke_width
        self.add_input('b_sy', 2.4, units='T', desc='flux density of stator yoke')
        self.add_output('w_sy', .005, units='m', desc='width of stator yoke')
        #self.declare_partials('w_sy', ['rot_or', 'b_g', 'n_m', 'k', 'b_sy'])

        # tooth_width
        self.add_input('b_t', 2.4, units='T', desc='flux density of tooth')
        self.add_input('n_s', 24, desc='Number of slots')
        self.add_output('w_t', 0.010, units='m', desc='width of tooth')
        #self.declare_partials('w_t', ['rot_or','b_g','n_s','k','b_t'])

        # slot_depth
        self.add_output('s_d', units='m', desc='slot depth')
        self.add_output('rot_ir', units='m', desc='rotor inner radius')
        self.add_output('sta_ir', units='m', desc='stator inner radius')

        # J
        self.add_input('n', 16, desc='number of wire turns')
        self.add_input('i', 30, units='A', desc='RMS current')
        self.add_input('k_wb', 0.65, desc='bare wire slot fill factor')
        self.add_output('j', units='A/mm**2', desc='Current density')

        self.declare_partials('*','*')#, method='fd')

    def compute(self,inputs,outputs):
        # rotor_outer_radius
        rot_or = inputs['rot_or']
        mot_or = inputs['mot_or']  # .0765
        gap = inputs['gap']
        # rotor_yoke_width
        b_g = inputs['b_g']
        n_m = inputs['n_m']
        k = inputs['k']
        b_ry = inputs['b_ry']
        t_mag = inputs['t_mag']

        n = inputs['n']
        i = inputs['i']
        k_wb = inputs['k_wb']
        outputs['w_ry'] = (pi*rot_or*b_g)/(n_m*k*b_ry)  
        # stator_yoke_width
        b_sy= inputs['b_sy']
        outputs['w_sy'] = (pi*rot_or*b_g)/(n_m*k*b_sy) 
        # tooth_width
        n_s = inputs['n_s']
        b_t = inputs['b_t']
        outputs['w_t'] = (2*pi*rot_or*b_g) / (n_s*k*b_t) 
        # Exec Comps
        # print(mot_or ,rot_or , gap , outputs['w_sy'])
        # print(mot_or - rot_or - gap - outputs['w_sy'])
        outputs['s_d'] = mot_or - rot_or - gap - outputs['w_sy']
        #outputs['mot_or'] = rot_or + gap + s_d + outputs['w_sy']
        outputs['rot_ir'] = (rot_or - t_mag) - outputs['w_ry'] 
        outputs['sta_ir'] = rot_or + gap
        area = pi*(mot_or-outputs['w_sy'])**2 - pi*(mot_or-outputs['w_sy']-outputs['s_d'])**2 #outputs['sta_ir']
        outputs['j'] = 2*n*i*(2.**0.5)/(k_wb/n_s*(area-n_s*1.25*(outputs['w_t']*outputs['s_d']))*1E6)
        print(outputs['j'])
        # TODO:  Better name for current density???

    # TODO: Get this partial working:
    # Use: check_partials function to check:
    def compute_partials(self, inputs, J):
    
        # rotor_yoke_width
        rot_or = inputs['rot_or']
        b_g= inputs['b_g']
        n_m= inputs['n_m']
        k = inputs['k']
        b_ry= inputs['b_ry']
        J['w_ry', 'rot_or'] = (pi*b_g)/(n_m*k*b_ry)
        J['w_ry', 'b_g'] = (pi*rot_or)/(n_m*k*b_ry)
        J['w_ry', 'n_m'] = -(pi*rot_or*b_g)/(n_m**2*k*b_ry)
        J['w_ry', 'k']   = -(pi*rot_or*b_g)/(n_m*k**2*b_ry)
        J['w_ry', 'b_ry'] = -(pi*rot_or*b_g)/(n_m*k*b_ry**2)
    
        # stator_yoke_width
        b_sy= inputs['b_sy']
        J['w_sy', 'rot_or'] = (pi*b_g)/(n_m*k*b_sy)
        J['w_sy', 'b_g'] = (pi*rot_or)/(n_m*k*b_sy)
        J['w_sy', 'n_m'] = -(pi*rot_or*b_g)/(n_m**2*k*b_sy)
        J['w_sy', 'k']   = -(pi*rot_or*b_g)/(n_m*k**2*b_sy)
        J['w_sy', 'b_sy'] = -(pi*rot_or*b_g)/(n_m*k*b_sy**2)
    
        # tooth_width
        n_s = inputs['n_s']
        b_t = inputs['b_t']
        J['w_t', 'rot_or'] = (2*pi*b_g)/(n_s*k*b_t)
        J['w_t', 'b_g'] = (2*pi*rot_or)/(n_s*k*b_t)
        J['w_t', 'n_s'] = -(2*pi*rot_or*b_g)/(n_s**2*k*b_t)
        J['w_t', 'k']   = -(2*pi*rot_or*b_g)/(n_s*k**2*b_t)
        J['w_t', 'b_t'] = -(2*pi*rot_or*b_g)/(n_s*k*b_t**2)
    
        # slot_depth
        mot_or = inputs['mot_or']
        gap = inputs['gap']
        J['s_d', 'mot_or'] = 1
        J['s_d', 'rot_or'] = -1 - J['w_sy', 'rot_or']
        J['s_d', 'gap'] = -1
        J['s_d', 'b_g'] = -J['w_sy', 'b_g']
        J['s_d', 'n_m'] = -J['w_sy', 'n_m']
        J['s_d', 'k'] = -J['w_sy', 'k']
        J['s_d', 'b_sy'] = -J['w_sy', 'b_sy']
    
        # rotor_inner_radius
        t_mag = inputs['t_mag']
        J['rot_ir', 'rot_or'] = 1 - J['w_ry', 'rot_or']
        J['rot_ir', 't_mag'] = -1
        J['rot_ir', 'b_g'] = - J['w_ry', 'b_g']
        J['rot_ir', 'n_m'] = - J['w_ry', 'n_m']
        J['rot_ir', 'k'] = - J['w_ry', 'k']
        J['rot_ir', 'b_ry'] = - J['w_ry', 'b_ry']
    
        # stator_inner_radius
        J['sta_ir', 'rot_or'] = 1
        J['sta_ir', 'gap'] = 1
    
        # current_density
        n = inputs['n']
        i = inputs['i']
        k_wb = inputs['k_wb']
        n_s = inputs['n_s']
        w_sy = (pi*rot_or*b_g)/(n_m*k*b_sy)
        s_d = mot_or - rot_or - gap - w_sy
        w_t = (2*pi*rot_or*b_g) / (n_s*k*b_t) 
        area = pi*(mot_or-w_sy)**2 - pi*(mot_or-w_sy-s_d)**2
    
        djdarea = -2*n*i*(2.**0.5)/(k_wb/n_s*((area-n_s*1.25*(w_t*s_d))**2)*1E6)
        djds_d = djdarea*(-n_s*1.25*w_t)
        djdw_t = djdarea*(-n_s*1.25*s_d)
        djdn_s = 2*n*i*(2.**0.5)*area/(k_wb*((area-n_s*1.25*(w_t*s_d))**2)*1E6)
    
        dads_d = 2*pi*(mot_or-w_sy-s_d)
        dadmot_or = 2*pi*s_d
        dadw_sy = -2*pi*s_d
    
    
        J['j', 'n'] = 2*i*(2.**0.5)/(k_wb/n_s*(area-n_s*1.25*(w_t*s_d))*1E6)
        J['j', 'i'] = 2*n*(2.**0.5)/(k_wb/n_s*(area-n_s*1.25*(w_t*s_d))*1E6)
        J['j', 'k_wb'] = - 2*n*i*(2.**0.5)/((k_wb**2)/n_s*(area-n_s*1.25*(w_t*s_d))*1E6)
        J['j', 'mot_or'] = djdarea*(dads_d*J['s_d', 'mot_or'] + dadmot_or) + djds_d*J['s_d', 'mot_or'] 
        J['j', 'rot_or'] = djdarea*(dadw_sy*J['w_sy', 'rot_or'] + dads_d*J['s_d','rot_or']) + djdw_t*J['w_t','rot_or'] + djds_d*J['s_d','rot_or']
        J['j', 'gap'] = (djdarea*dads_d + djds_d)*J['s_d','gap']
        J['j', 'b_g'] = djdarea*(dadw_sy*J['w_sy','b_g']+dads_d*J['s_d','b_g']) + djdw_t*J['w_t','b_g'] + djds_d*J['s_d','b_g']
        J['j', 'n_m'] = djdarea*(dadw_sy*J['w_sy','n_m']+dads_d*J['s_d','n_m']) + djds_d*J['s_d','n_m']
        J['j', 'n_s'] = djdn_s + djdw_t*J['w_t','n_s']
        J['j', 'k'] = djdarea*(dadw_sy*J['w_sy','k'] + dads_d*J['s_d','k']) + djdw_t*J['w_t','k'] + djds_d*J['s_d','k']
        J['j', 'b_sy'] = djdarea*(dadw_sy*J['w_sy','b_sy'] + dads_d*J['s_d','b_sy']) + djds_d*J['s_d','b_sy']
        J['j', 'b_t'] = djdw_t*J['w_t','b_t']

class torque(ExplicitComponent):

    def setup(self):
       self.add_input('b_g', 2.4, units='T', desc='air gap flux density')    
       self.add_input('n_m', 16, desc='number of poles')
       self.add_input('n', 16, desc='number of wire turns')
       self.add_input('l_st', .045, units='m', desc='stack length')
       self.add_input('i', 30, units='A', desc='RMS current')       
       self.add_input('rot_or', .025, units='m', desc='rotor outer radius')
       self.add_output('tq', 25, units='N*m', desc='torque')
       #self.declare_partials('tq', ['n_m','n','b_g','l_st','rot_or','i'])
       self.declare_partials('*','*')#, method='fd')

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
    
       J['tq', 'n_m'] = 2*n*b_g*l_st*rot_or*i*.68
       J['tq', 'n'] = 2*n_m*b_g*l_st*rot_or*i*.68
       J['tq', 'b_g'] = 2*n_m*n*l_st*rot_or*i*.68
       J['tq', 'l_st'] = 2*n_m*n*b_g*rot_or*i*.68
       J['tq', 'rot_or'] = 2*n_m*n*b_g*l_st*i*.68
       J['tq', 'i'] = 2*n_m*n*b_g*l_st*rot_or*.68

class motor_mass(ExplicitComponent):

    def setup(self):
        # stator
        self.add_input('rho', 8110.2, units='kg/m**3', desc='density of hiperco-50')
        self.add_input('mot_or', .075, units='m', desc='motor outer radius')           
        self.add_input('n_s', 15, desc='number of slots')                           
        self.add_input('sta_ir', .050, units='m', desc='stator inner radius')       
        self.add_input('w_t', units='m', desc='tooth width')                        
        self.add_input('l_st', units='m', desc='length of stack')  
        self.add_input('s_d', units='m', desc='slot depth')                 
        self.add_output('sta_mass', 25, units='kg', desc='mass of stator')
        #self.declare_partials('sta_mass', ['rho','mot_or','n_s','sta_ir','w_t','l_st','s_d'], method='fd')
        # rotor
        self.add_input('rot_or', 0.0615, units='m', desc='rotor outer radius')
        self.add_input('rot_ir', 0.0515, units='m', desc='rotor inner radius')
        self.add_input('t_mag', .005, units='m', desc='magnet thickness')
        self.add_output('rot_mass', 1.0, units='kg', desc='weight of rotor')
        #magnets
        self.add_input('rho_mag', 7500, units='kg/m**3', desc='density of magnet')
        self.add_output('mag_mass', 0.5, units='kg', desc='mass of magnets')
        
        #self.declare_partials('rot_mass',['rho','rot_or','rot_ir'], method='fd')
        self.declare_partials('*','*')#, method='fd')

    def compute(self,inputs,outputs):
        # stator
        rho=inputs['rho']
        mot_or=inputs['mot_or']
        n_s=inputs['n_s']
        sta_ir=inputs['sta_ir']
        w_t=inputs['w_t']
        l_st=inputs['l_st']
        s_d=inputs['s_d']

        outputs['sta_mass'] = rho * l_st * ((pi * mot_or**2)-(pi * (sta_ir+s_d)**2)+(n_s*(w_t*s_d*1.5)))
        
        # rotor
        rot_ir=inputs['rot_ir']
        rot_or=inputs['rot_or']
        l_st=inputs['l_st']
        t_mag=inputs['t_mag']
        print('rot_or: ',rot_or)
        outputs['rot_mass'] = (pi*(rot_or - t_mag)**2 - pi*rot_ir**2) * rho * l_st

        # magnets
        rho_mag=inputs['rho_mag']
        outputs['mag_mass'] = (((pi*rot_or**2) - (pi*(rot_or-t_mag)**2))) * rho_mag * l_st

        #


    def compute_partials(self,inputs,J):
    
        #stator
        rho=inputs['rho']
        mot_or=inputs['mot_or']
        n_s=inputs['n_s']
        sta_ir=inputs['sta_ir']
        w_t=inputs['w_t']
        l_st=inputs['l_st']
        s_d=inputs['s_d']
    
        J['sta_mass', 'rho'] = l_st * ((pi * mot_or**2)-(pi * (sta_ir+s_d)**2)+(n_s*(w_t*s_d*1.5)))
        J['sta_mass', 'mot_or'] = 2 * rho * l_st * (pi * mot_or)
        J['sta_mass', 'n_s'] = rho * l_st * (w_t*s_d*1.5)
        J['sta_mass', 'sta_ir'] = 2 * rho * l_st * -(pi * (sta_ir+s_d))
        J['sta_mass', 'w_t'] = rho * l_st * (n_s*(s_d*1.5))
        J['sta_mass', 'l_st'] = rho * ((pi * mot_or**2)-(pi * (sta_ir+s_d)**2)+(n_s*(w_t*s_d*1.5)))
        J['sta_mass', 's_d'] = rho * l_st * (-(2 * pi * (sta_ir+s_d)) + (n_s*w_t*1.5))
    
        #rotor
        rot_ir=inputs['rot_ir']
        rot_or=inputs['rot_or']
        l_st=inputs['l_st']
        t_mag=inputs['t_mag']    
    
        J['rot_mass', 'rot_ir'] = -(2 * pi * rot_ir) * rho * l_st
        J['rot_mass', 'rot_or'] = (2 * pi * (rot_or - t_mag)) * rho * l_st
        J['rot_mass', 'l_st'] = (pi*(rot_or - t_mag)**2 - pi*rot_ir**2) * rho
        J['rot_mass', 't_mag'] = (2 * pi * (rot_or - t_mag)) * rho * l_st
        J['rot_mass', 'rho'] = (pi*(rot_or - t_mag)**2 - pi*rot_ir**2) * l_st
    
        #magnets
        rho_mag=inputs['rho_mag']
    
        J['mag_mass', 'rot_or'] = ((2*pi*rot_or) - (2*pi*(rot_or-t_mag))) * rho_mag * l_st
        J['mag_mass', 't_mag'] = - (2*pi*(rot_or-t_mag)) * rho_mag * l_st
        J['mag_mass', 'rho_mag'] = (((pi*rot_or**2) - (pi*(rot_or-t_mag)**2))) * l_st
        J['mag_mass', 'l_st'] = (((pi*rot_or**2) - (pi*(rot_or-t_mag)**2))) * rho_mag
    
    

if __name__ == "__main__":
    p = Problem()
    model = p.model

    ind = model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])

    #ind.add_output('rot_or', val=0.0615, units='m')         # Outer radius of rotor, including 5mm magnet thickness
    ind.add_output('k', val=0.97)                   # Stacking factor
    ind.add_output('k_wb', val=0.55)                # copper fill factor
    ind.add_output('gap', val=0.001, units='m')     # air gap
    ind.add_output('n', val=24)                     # Number of wire turns     
    ind.add_output('i', val=33, units='A')          # RMS Current
    ind.add_output('mot_or', val=0.08, units='m')    # Motor outer radius
    ind.add_output('t_mag', val=.005, units='m')    # Magnet thickness

    ind.add_output('b_g', val = 1, units='T')         # Air gap flux Density    !! Flux values may represent 100% slot fill !!
    ind.add_output('b_ry', val=4, units='T')          # Rotor yoke flux density
    ind.add_output('b_sy', val=4, units='T')          # Stator yoke flux density
    ind.add_output('b_t', val=4, units='T')           # Tooth Flux Density

    ind.add_output('n_s', val=21)                # Number of Slots
    ind.add_output('n_m', val=20)                # Number of poles

    #ind.add_output('l_st', val=0.033, units='m')         # Stack Length
    ind.add_output('rho', val=8110.2, units='kg/m**3')   # Density of Hiperco-50
    ind.add_output('rho_mag', val=7500, units='kg/m**3')    # Density of Magnets

    ind.add_output('rot_or', val = 0.06, units='m')
    ind.add_output('l_st', val = 0.02, units='m')

    # bal = BalanceComp()
    # 
    # bal.add_balance('rot_or', val=0.06, units='m', use_mult=False, rhs_val = 13.)
    # bal.add_balance('l_st', val=0.02, units='m', use_mult=False, rhs_val = 24.)

    model.add_subsystem('size', motor_size(), promotes_inputs=['n','i','k_wb','mot_or','gap','rot_or','b_g','k','b_ry','n_m','b_sy','b_t','n_s'], promotes_outputs=['w_ry', 'w_sy', 'w_t','s_d','rot_ir','sta_ir'])
    # model.add_subsystem('motor_radius_prime', ExecComp('r_m_p = rot_or + .005 + .001 + s_d + w_sy',r_m_p={'units':'m'}, rot_or={'units':'m'}, s_d={'units':'m'}, w_sy={'units':'m'}), promotes_inputs=['rot_or','s_d','w_sy'], promotes_outputs=['r_m_p'])
    # model.add_subsystem('mass_stator', mass_stator(), promotes_inputs=['rho','mot_or','n_s','sta_ir','w_t','l_st'], promotes_outputs=['weight']
    # model.add_subsystem('stmass', ExecComp('mass = l_st * ((pi * mot_or**2)-(pi * sta_ir**2)+(n_s*(w_t*1.2)))', l_st={'units':'m'},mot_or={'units':'m'},sta_ir={'units':'m'},w_t={'units':'m'}), promotes_inputs=['l_st','mot_or','sta_ir','n_s','w_t'], promotes_outputs=['mass']
    # model.add_subsystem(name='balance', subsys=bal)
    model.add_subsystem('mass', motor_mass(), promotes_inputs=['t_mag','rho_mag','rho','mot_or','n_s','sta_ir','w_t','l_st','s_d','rot_or','rot_ir'], promotes_outputs=['sta_mass','rot_mass','mag_mass'])
    model.add_subsystem('torque', torque(), promotes_inputs=['rot_or','b_g','i','n_m','n','l_st'], promotes_outputs=['tq'])
    model.add_subsystem('efficiency', efficiency(), promotes_inputs=['tq','i'], promotes_outputs=['nu'])

    # model.connect('balance.rot_or', 'rot_or')
    # model.connect('size.j', 'balance.lhs:rot_or')
    # 
    # model.connect('balance.l_st', 'l_st')
    # model.connect('tq', 'balance.lhs:l_st')

    #model.linear_solver = DirectSolver()

    #model.nonlinear_solver = NewtonSolver()
    #model.nonlinear_solver.options['maxiter'] = 100
    #model.nonlinear_solver.options['iprint'] = 2
    
    #optimization setup
    p.driver = ScipyOptimizeDriver()
    p.driver.options['optimizer'] = 'SLSQP'
    p.driver.options['debug_print'] = ['desvars', 'objs']
    model.add_objective('nu', ref=-1)
    model.add_design_var('gap', lower=.001, upper=.003)
    model.add_design_var('mot_or', lower=.07, upper=.10)
    model.add_design_var('rot_or', lower = .05, upper=.068)
    model.add_design_var('l_st', lower = .0004, upper = .003)
    model.add_constraint('tq', lower=20, scaler=1)
    model.add_constraint('size.j', upper=15, scaler=1)
    
    p.setup()
    
    p['gap'] = 0.002
    p['mot_or'] = 0.09
    p['rot_or'] = 0.06
    p['l_st'] = 0.01
    
    p.final_setup()
    #p.check_partials(compact_print=True)
    p.model.list_outputs(implicit=True)
    p.set_solver_print(level=2)
    #view_model(p)
    #p.run_model()
    p.run_driver()
    
    print('Rotor Outer Radius................',  p.get_val('rot_or', units='mm'))
    print('Rotor Inner Radius................',  p.get_val('rot_ir', units='mm'))
    # print('Rotor Outer Radius................',  p.get_val('rot_or', units='mm'))

    print('Stator Inner Radius...............',  p.get_val('sta_ir', units='mm'))
    print('Motor Outer Radius................',  p.get_val('mass.mot_or', units='mm'))

    print('Rotor Yoke Thickness..............',  p.get_val('w_ry', units='mm'))
    print('Slot Depth........................',  p.get_val('s_d', units='mm'))
    print('Stator Yoke Thickness.............',  p.get_val('w_sy', units='mm'))
    print('Tooth Width.......................',  p.get_val('w_t', units='mm'))
    print('Magnet Thickness..................',  p.get_val('t_mag', units='mm'))

    print('Torque............................',  p['tq'])

    print('Mass of Stator....................',  p.get_val('sta_mass', units='kg'))
    print('Mass of Rotor.....................',  p.get_val('rot_mass', units='kg'))
    print('Mass of Magnets...................',  p.get_val('mag_mass', units='kg'))    
    print('Current Density...................',  p.get_val('size.j'))
    print('Stack Length......................',  p.get_val('mass.l_st', units='mm'))
    
    print('Efficiency........................',  p.get_val('nu'))

    from solid import *
    from solid.utils import *

    mot_or = float(p.get_val('mass.mot_or', units='mm'))
    l_st = float(p.get_val('mass.l_st', units='mm'))
    sta_ir = float(p.get_val('sta_ir', units='mm'))
    w_sy = float(p.get_val('w_sy', units='mm'))
    w_ry = float(p.get_val('w_ry', units='mm'))
    w_t = float(p.get_val('w_t', units='mm'))
    s_d = float(p.get_val('s_d', units='mm'))
    n_s = float(p.get_val('n_s'))
    n_m = float(p.get_val('n_m') )
    fa = 1
    fs = 0.05

    rot_ir = float(p.get_val('rot_ir', units='mm'))
    rot_or = float(p.get_val('rot_or', units='mm'))

    stator_yolk = cylinder(r=mot_or, h=l_st, center=True) - cylinder(r=mot_or-w_sy, h=l_st+1, center=True)
    slot = cube([s_d, w_t, l_st], center=True)
    rotor = color("Blue");cylinder(r=rot_or, h=l_st, center=True) - cylinder(r=rot_ir, h=l_st+1, center=True)

    print(scad_render(stator_yolk+slot+rotor, file_header='$fa = %s; $fs = %s;' % (fa, fs)))


'''
$fa = 1; $fs = 0.05;

union() {
    difference() {
        cylinder(center = true, h = 33.0000000000, r = 79.5000000000);
        cylinder(center = true, h = 34.0000000000, r = 76.9445168375);
    }
    
    difference() {
        color("blue")cylinder(center = true, h = 33.0000000000, r = 63.1225990365);
        cylinder(center = true, h = 34.0000000000, r = 55.5671158741);
    }
}


module slot2(){
    Stator_IR = 64;
    Tooth_angle = 20;
    Stator_slots = 16;
    Tooth_width = 4.86;
    Tooth_tip = 3;
    Tooth_angle = 20;
    Stator_OR = 73.;
    Stator_yoke = 5;
    Tooth_point_angle = 45;

    points = [
    [Stator_IR*cos(Tooth_angle/2),             -Stator_IR*sin(Tooth_angle/2)],
    [(Stator_IR+Tooth_tip)*cos(Tooth_angle/2), -(Stator_IR+Tooth_tip)*sin(Tooth_angle/2)],
    [-tan(Tooth_point_angle+180/Stator_slots)*(Tooth_width/2-(Stator_IR+Tooth_tip)*sin(Tooth_angle/2))+(Stator_IR+Tooth_tip)*cos(Tooth_angle/2),  -Tooth_width/2],
    [Stator_OR-Stator_yoke,                    -Tooth_width/2], 
    [Stator_OR-Stator_yoke,                    Tooth_width/2],
    [-tan(Tooth_point_angle+180/Stator_slots)*(Tooth_width/2-(Stator_IR+Tooth_tip)*sin(Tooth_angle/2))+(Stator_IR+Tooth_tip)*cos(Tooth_angle/2),  Tooth_width/2],
    [(Stator_IR+Tooth_tip)*cos(Tooth_angle/2),    (Stator_IR+Tooth_tip)*sin(Tooth_angle/2)],
    [Stator_IR*cos(Tooth_angle/2),    Stator_IR*sin(Tooth_angle/2)]
    ];

    linear_extrude(height = 20, center = true, convexity = 10, twist = 0)
    polygon(points = points, paths = [[5,4,6,7,0,1,3,2]], convexity = 4);
}

difference() {
    for (i=[0:30:360]) rotate(i) slot2();
    cylinder(center = true, h = 34.0000000000, r = 65.9445168375);
}


'''
