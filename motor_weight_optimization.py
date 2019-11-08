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

class motor_mass(ExplicitComponent):

    def setup(self):
        # stator
        self.add_input('rho', 8110.2, units='kg/m**3', desc='density of hiperco-50')
        self.add_input('radius_motor', .075, units='m', desc='motor outer radius')           
        self.add_input('n_s', 15, desc='number of slots')                           
        self.add_input('sta_ir', .050, units='m', desc='stator inner radius')       
        self.add_input('w_t', units='m', desc='tooth width')                        
        self.add_input('stack_length', units='m', desc='length of stack')  
        self.add_input('s_d', units='m', desc='slot depth')                 
        self.add_output('sta_mass', 25, units='kg', desc='mass of stator')
        #self.declare_partials('sta_mass', ['rho','radius_motor','n_s','sta_ir','w_t','stack_length','s_d'], method='fd')
        # rotor
        self.add_input('rot_or', 0.0615, units='m', desc='rotor outer radius')
        self.add_input('rot_ir', 0.0515, units='m', desc='rotor inner radius')
        self.add_input('t_mag', .005, units='m', desc='magnet thickness')
        self.add_output('rot_mass', 1.0, units='kg', desc='weight of rotor')
        #magnets
        self.add_input('rho_mag', 7500, units='kg/m**3', desc='density of magnet')
        self.add_output('mag_mass', 0.5, units='kg', desc='mass of magnets')
        
        #self.declare_partials('rot_mass',['rho','rot_or','rot_ir'], method='fd')
        self.declare_partials('*','*', method='fd')

    def compute(self,inputs,outputs):
        # stator
        rho=inputs['rho']
        radius_motor=inputs['radius_motor']
        n_s=inputs['n_s']
        sta_ir=inputs['sta_ir']
        w_t=inputs['w_t']
        stack_length=inputs['stack_length']
        s_d=inputs['s_d']

        outputs['sta_mass'] = rho * stack_length * ((pi * radius_motor**2)-(pi * (sta_ir+s_d)**2)+(n_s*(w_t*s_d*1.5)))
        
        # rotor
        rot_ir=inputs['rot_ir']
        rot_or=inputs['rot_or']
        stack_length=inputs['stack_length']
        t_mag=inputs['t_mag']
        # print('rot_or: ',rot_or)
        outputs['rot_mass'] = (pi*(rot_or - t_mag)**2 - pi*rot_ir**2) * rho * stack_length

        # magnets
        rho_mag=inputs['rho_mag']
        outputs['mag_mass'] = (((pi*rot_or**2) - (pi*(rot_or-t_mag)**2))) * rho_mag * stack_length

        #


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

# ---------------------------------------------------------
# -------------CORE LOSSES---------------------------------
# ---------------------------------------------------------
# Need to have Bpk change with the actual field density
class motor_losses(ExplicitComponent):

    def setup(self):
        self.add_input('K_h', 0.0073790325365744, desc='Hysteresis constant for 0.006in Hiperco-50')
        self.add_input('K_e', 0.00000926301369333214, desc='Eddy constant for 0.006in Hiperco-50')
        self.add_input('f_e', 1000, units='Hz', desc='Electrical frequency')
        self.add_input('K_h_alpha', 1.15293258569149, desc='Hysteresis constant for steinmetz alpha value')
        self.add_input('K_h_beta', 1.72240393990502, desc='Hysteresis constant for steinmetz beta value')
        self.add_input('B_pk', 2.4, units='T', desc='Peak magnetic field in Tesla')

        self.add_output('P_h', units='W', desc='Core hysteresis losses')
        self.add_output('P_e', units='W', desc='Core eddy current losses')
    
    def compute(self,inputs,outputs):
        K_h=inputs['K_h']
        K_e=inputs['K_e']
        K_h_alpha=inputs['K_h_alpha']
        K_h_beta=inputs['K_h_beta']
        B_pk=inputs['B_pk']
        f_e=inputs['f_e']

        outputs['P_h'] = K_h*f_e*B_pk**(K_h_alpha+K_h_beta*B_pk)
        outputs['P_e'] = K_e*f_e**2*B_pk**2

    # def compute_partials(self,inputs,J):


# ---------------------------------------        

if __name__ == "__main__":
    p = Problem()
    model = p.model

    ind = model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])

    ind.add_output('k', val=0.94)                   # Stacking factor
    ind.add_output('k_wb', val=0.50)                # copper fill factor
    ind.add_output('gap', val=0.001, units='m')     # Air gap distance
    ind.add_output('n', val=12)                     # Number of wire turns     
    ind.add_output('i', val=34.8, units='A')        # RMS Current
    ind.add_output('radius_motor', val=0.0795, units='m')    # Motor outer radius
    ind.add_output('t_mag', val=.0044, units='m')   # Magnet thickness

    ind.add_output('b_g', val = 1, units='T')           # Air gap flux Density    !! Flux values may represent 100% slot fill !!
    ind.add_output('b_ry', val=3, units='T')          # Rotor yoke flux density
    ind.add_output('b_sy', val=3, units='T')          # Stator yoke flux density
    ind.add_output('b_t', val=3, units='T')           # Tooth Flux Density

    ind.add_output('n_s', val=21)                # Number of Slots
    ind.add_output('n_m', val=20)                # Number of poles

    ind.add_output('stack_length', val=0.0345, units='m')           # Stack Length
    ind.add_output('rho', val=8110.2, units='kg/m**3')      # Density of Hiperco-50
    ind.add_output('rho_mag', val=7500, units='kg/m**3')    # Density of Magnets

    ind.add_output('K_h', val=0.0073790325365744)
    ind.add_output('K_e', val=0.00000926301369333214)
    ind.add_output('f_e', val=277, units='Hz')              # Frequency: 5000 RPM / 60 - sec / 3 - phases
    ind.add_output('K_h_alpha', val=1.15293258569149)
    ind.add_output('K_h_beta', val=1.72240393990502)
    ind.add_output('B_pk', val=2.4, units='T')

    bal = BalanceComp()
    bal.add_balance('rot_or', val=0.06, units='m', use_mult=False)
    tgt = IndepVarComp(name='J', val=10, units='A/mm**2')
    model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['J'])
    
    model.add_subsystem('size', motor_size(), promotes_inputs=['n','i','k_wb','radius_motor','gap','rot_or','b_g','k','b_ry','n_m','b_sy','b_t','n_s'], promotes_outputs=['w_ry', 'w_sy', 'w_t','s_d','rot_ir','sta_ir'])
    model.add_subsystem('losses', motor_losses(), promotes_inputs=['f_e', 'B_pk'], promotes_outputs=['P_e','P_h'])
    model.add_subsystem(name='balance', subsys=bal)
    model.add_subsystem('mass', motor_mass(), promotes_inputs=['t_mag','rho_mag','rho','radius_motor','n_s','sta_ir','w_t','stack_length','s_d','rot_or','rot_ir'], promotes_outputs=['sta_mass','rot_mass','mag_mass'])
    model.add_subsystem('torque', torque(), promotes_inputs=['rot_or','b_g','i','n_m','n','stack_length'], promotes_outputs=['tq'])

    model.connect('J', 'balance.rhs:rot_or')
    model.connect('balance.rot_or', 'rot_or')
    model.connect('size.J', 'balance.lhs:rot_or')

    model.linear_solver = DirectSolver()

    model.nonlinear_solver = NewtonSolver()
    model.nonlinear_solver.options['maxiter'] = 100
    model.nonlinear_solver.options['iprint'] = 0

    p.setup()
    p.final_setup()
    #p.check_partials(compact_print=True)
    # p.model.list_outputs(implicit=True)
    # p.set_solver_print(2)
    #view_model(p)
    p.run_model()

    print('Rotor Inner Diameter..............', 2 * p.get_val('rot_ir', units='mm'))

    print('Stator Inner Diameter.............', 2 *  p.get_val('sta_ir', units='mm'))
    print('Motor Outer Diameter..............', 2 *  p.get_val('mass.radius_motor', units='mm'))

    print('Rotor Yoke Thickness..............',  p.get_val('w_ry', units='mm'))
    print('Slot Depth........................',  p.get_val('s_d', units='mm'))
    print('Stator Yoke Thickness.............',  p.get_val('w_sy', units='mm'))
    print('Tooth Width.......................',  p.get_val('w_t', units='mm'))
    print('Magnet Thickness..................',  p.get_val('t_mag', units='mm'))

    print('Torque............................',  p['tq'])

    print('Mass of Stator....................',  p.get_val('sta_mass', units='kg'))
    print('Mass of Rotor.....................',  p.get_val('rot_mass', units='kg'))
    print('Mass of Magnets...................',  p.get_val('mag_mass', units='kg'))    
    print('Current Density...................',  p.get_val('size.J'))
    print('Core Eddy Losses .................',  p.get_val('P_e'))
    print('Core Hysteresis Losses ...........',  p.get_val('P_h'))

