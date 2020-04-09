import numpy as np
from math import pi

import openmdao.api as om

class MotorSizeComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('radius_motor', 0.078225, units='m', desc='outer radius of motor')
        self.add_input('gap', 0.001, units='m', desc='air gap')
        self.add_input('rot_or', .05, units='m', desc='rotor outer radius') 
        self.add_input('B_g', 1.0, units='T', desc='air gap flux density')
        self.add_input('k', 0.95, desc='stacking factor')
        self.add_input('b_ry', 2.4, units='T', desc='flux density of rotor yoke')
        self.add_input('n_m', 20, desc='number of poles')
        self.add_input('t_mag', 0.0045, units='m', desc='magnet thickness')
        self.add_input('b_sy', 2.4, units='T', desc='flux density of stator yoke')
        self.add_input('b_t', 2.4, units='T', desc='flux density of tooth')
        self.add_input('n_slots', 20, desc='Number of slots')
        self.add_input('n_turns', 11, desc='number of wire turns')
        self.add_input('I_required',30*np.ones(nn), units='A', desc='RMS current')  # Imax
        self.add_input('k_wb', 0.65, desc='bare wire slot fill factor')

        self.add_output('w_ry', .004, units='m', desc='width of stator yoke')
        self.add_output('w_sy', .005, units='m', desc='width of stator yoke')
        self.add_output('w_t', 0.0048, units='m', desc='width of tooth')   
        self.add_output('s_d', .012, units='m', desc='slot depth')
        self.add_output('rot_ir', .061, units='m', desc='rotor inner radius')
        self.add_output('sta_ir', .070, units='m', desc='stator inner radius')
        self.add_output('slot_area', 0.0002, units='m**2', desc='area of one slot')

        self.add_output('w_slot', .015, units='m', desc='width of a slot')
        self.add_output('J',  units='A/mm**2', desc='Current density')

        r = c = np.arange(nn) 
        self.declare_partials('w_ry', ['rot_or', 'B_g', 'n_m', 'k', 'b_ry'])
        self.declare_partials('w_sy', ['rot_or', 'B_g', 'n_m', 'k', 'b_sy'])
        self.declare_partials('w_t', ['rot_or','B_g','n_slots','k','b_t'])
        self.declare_partials('s_d', ['radius_motor', 'rot_or', 'gap', 'B_g', 'n_m', 'k', 'b_sy'])
        self.declare_partials('rot_ir', ['rot_or', 't_mag', 'B_g', 'n_m', 'k', 'b_ry'])
        self.declare_partials('sta_ir', ['rot_or', 'gap'])
        self.declare_partials('slot_area', ['n_slots', 'radius_motor', 'rot_or', 'B_g', 'n_m', 'k', 'b_sy', 'gap', 'b_t'])
        self.declare_partials('w_slot', ['n_slots', 'radius_motor', 'rot_or', 'B_g', 'n_m', 'k', 'b_sy', 'gap', 'b_t'])
        self.declare_partials(of='J', wrt='I_required', rows=r, cols=c)
        self.declare_partials('J', ['n_turns', 'k_wb', 'n_slots', 'radius_motor', 'rot_or', 'B_g', 'n_m', 'k', 'b_sy', 'gap', 'b_t'])

    def compute(self,inputs,outputs):
        radius_motor = inputs['radius_motor']
        gap = inputs['gap']
        B_g = inputs['B_g']
        n_m = inputs['n_m']
        k = inputs['k']
        b_ry = inputs['b_ry']
        t_mag = inputs['t_mag']
        n_turns = inputs['n_turns']
        I = inputs['I_required']
        k_wb = inputs['k_wb']
        b_sy= inputs['b_sy']
        n_slots = inputs['n_slots']
        b_t = inputs['b_t']
        rot_or = inputs['rot_or']

        outputs['w_ry'] = (pi*rot_or*B_g)/(n_m*k*b_ry) 
        outputs['w_t'] = (2*pi*rot_or*B_g) / (n_slots*k*b_t) 
        outputs['w_sy'] = (pi*rot_or*B_g)/(n_m*k*b_sy)
        outputs['s_d'] = radius_motor - rot_or - gap - outputs['w_sy']
        outputs['rot_ir'] = rot_or - t_mag - outputs['w_ry'] 
        outputs['sta_ir'] = rot_or + gap
        outputs['slot_area'] = (pi*(radius_motor-outputs['w_sy'])**2 - \
                               pi*(radius_motor-outputs['w_sy']-outputs['s_d'])**2)/n_slots - (outputs['w_t'] * outputs['s_d']*1.05)
        outputs['w_slot'] = outputs['slot_area'] / outputs['s_d']
        outputs['J'] = 2*n_turns*I*(2.**0.5)/(k_wb*outputs['slot_area']*1E6)

    def compute_partials(self, inputs, J):

        radius_motor = inputs['radius_motor']
        gap = inputs['gap']
        B_g = inputs['B_g']
        n_m = inputs['n_m']
        k = inputs['k']
        b_ry = inputs['b_ry']
        t_mag = inputs['t_mag']
        n_turns = inputs['n_turns']
        I = inputs['I_required']
        k_wb = inputs['k_wb']
        b_sy= inputs['b_sy']
        n_slots = inputs['n_slots']
        b_t = inputs['b_t']
        rot_or = inputs['rot_or']

        slot_area = pi/n_slots*((radius_motor**2 - (2*radius_motor*pi*rot_or*B_g)/(n_m*k*b_sy) + ((pi*rot_or*B_g)/(n_m*k*b_sy))**2) -  (rot_or**2 + 2*rot_or*gap + gap**2 ))  \
         - ( 1.05*(2*pi*rot_or*B_g)/(n_slots*k*b_t) * (radius_motor - rot_or - gap - (pi*rot_or*B_g)/(n_m*k*b_sy)) )
        w_ry = (pi*rot_or*B_g)/(n_m*k*b_ry) 
        w_t = (2*pi*rot_or*B_g) / (n_slots*k*b_t) 
        w_sy = (pi*rot_or*B_g)/(n_m*k*b_sy)
        s_d = radius_motor - rot_or - gap - w_sy
        rot_ir = (rot_or- t_mag) - w_ry 
        sta_ir = rot_or + gap
        w_slot = slot_area / s_d
        J_a = 2*n_turns*I*(2.**0.5)/(k_wb*(slot_area)*1E6)

        # intermediate calculation used to make derivatives easier 
        gamma = pi/n_slots * (radius_motor-w_ry)**2 - (radius_motor-w_sy-s_d)**2

        dw_sy__drot_or = pi*B_g/(n_m*k*b_sy)
        ds_d__drot_or = -1 - dw_sy__drot_or
        dw_t__drot_or = 2*pi*B_g/(n_slots*k*b_t)
        dgamma__drot_or = -2*pi/n_slots*((radius_motor - w_sy)*dw_sy__drot_or - 
                                         (radius_motor - w_sy - s_d)*(dw_sy__drot_or + ds_d__drot_or))
        dslot_area__drot_or = dgamma__drot_or - ds_d__drot_or * w_t*1.05 - s_d* dw_t__drot_or*1.05
        dw_ry__drot_or = pi*B_g/(n_m*k*b_ry)
        ds_d__drot_or = -1 - dw_sy__drot_or

        dw_sy__dB_g = pi*rot_or/(n_m*k*b_sy)
        ds_d__dB_g = -dw_sy__dB_g
        dw_t__dB_g = 2*pi*rot_or/(n_slots*k*b_t)
        dgamma__dB_g = -2*pi/n_slots*((radius_motor-w_sy)*(dw_sy__dB_g) - (radius_motor - w_sy - s_d)*(dw_sy__dB_g+ds_d__dB_g))
        dw_ry__dB_g = pi*rot_or/(n_m*k*b_ry)
        dslot_area__dBg = dgamma__dB_g - ds_d__dB_g * w_t*1.05 - s_d* dw_t__dB_g*1.05

        dw_t__dn_slots = -2*pi*rot_or*B_g/(n_slots**2*k*b_t)
        dgamma__dn_slots = -pi/n_slots**2*(2*radius_motor*s_d - s_d**2 - 2*s_d*w_sy)
        ds_d__dn_slots = -pi/n_slots**2*((radius_motor-w_ry)**2 - (radius_motor-w_sy-s_d)**2)
        dslot_area__dn_slots = (- pi/n_slots**2 * ((radius_motor - w_sy)**2 - (radius_motor - w_sy - s_d)**2) 
                                - dw_t__dn_slots * s_d * 1.05) 

        dw_sy__dn_m = -B_g*pi*rot_or/(n_m**2*k*b_sy)
        ds_d__dn_m = -dw_sy__dn_m
        dgamma__dn_m = -2*pi/n_slots*((radius_motor-w_sy)*(dw_sy__dn_m) - (radius_motor - w_sy - s_d)*(dw_sy__dn_m+ds_d__dn_m))
        dw_ry__dn_m = -pi*rot_or*B_g/(k*n_m**2*b_ry)
        dslot_area__dn_m =  dgamma__dn_m - ds_d__dn_m*w_t*1.05

        dw_sy__dk = -pi*rot_or*B_g/(n_m*k**2*b_sy)
        ds_d__dk = -dw_sy__dk
        dw_t__dk = -2*pi*rot_or*B_g/(n_slots*k**2*b_t)
        dgamma__dk = -2*pi/n_slots*((radius_motor-w_sy)*dw_sy__dk - ((radius_motor - w_sy - s_d)*(dw_sy__dk+ds_d__dk) ))
        dw_ry__dk = -pi*rot_or*B_g/(k**2*n_m*b_ry)
        dslot_area__dk = dgamma__dk - ds_d__dk*w_t*1.05 - s_d*dw_t__dk*1.05

        dw_sy__db_sy = -pi*rot_or*B_g/(n_m*k*b_sy**2)
        ds_d__db_sy = -dw_sy__db_sy
        dgamma__db_sy = -2*pi/n_slots*((radius_motor-w_sy)*(dw_sy__db_sy) - (radius_motor - w_sy - s_d)*(dw_sy__db_sy+ds_d__db_sy))
        dslot_area__db_sy = dgamma__db_sy - ds_d__db_sy*w_t *1.05

        ds_d__dgap   = -gap**-2
        dgamma__dgap = -2*pi/n_slots*(radius_motor-w_sy-s_d)
        dslot_area__dgap = dgamma__dgap + w_t*1.05

        dw_ry__db_ry = -pi*rot_or*B_g/(k*n_m*b_ry**2)
        
        dw_t__db_t = -2*pi*rot_or*B_g/(n_slots*k*b_t**2)
        dslot_area__db_t = -s_d*dw_t__db_t*1.05

        dslot_area__dradius_motor = 2*pi/n_slots * (radius_motor - w_sy) - w_t*1.05



        # Rotor Yoke Width
        J['w_ry', 'rot_or'] = (pi*B_g)/(n_m*k*b_ry)
        J['w_ry', 'B_g'] = (pi*rot_or)/(n_m*k*b_ry)
        J['w_ry', 'n_m'] = -(pi*rot_or*B_g)/(n_m**2*k*b_ry)
        J['w_ry', 'k']   = -(pi*rot_or*B_g)/(n_m*k**2*b_ry)
        J['w_ry', 'b_ry'] = -(pi*rot_or*B_g)/(n_m*k*b_ry**2)

        # Stator Yoke Width
        J['w_sy', 'rot_or'] = (pi*B_g)/(n_m*k*b_sy)
        J['w_sy', 'B_g'] = (pi*rot_or)/(n_m*k*b_sy)
        J['w_sy', 'n_m'] = -(pi*rot_or*B_g)/(n_m**2*k*b_sy)
        J['w_sy', 'k']   = -(pi*rot_or*B_g)/(n_m*k**2*b_sy)
        J['w_sy', 'b_sy'] = -(pi*rot_or*B_g)/(n_m*k*b_sy**2)

        # Tooth Width
        J['w_t', 'rot_or'] = (2*pi*B_g)/(n_slots*k*b_t)
        J['w_t', 'B_g'] = (2*pi*rot_or)/(n_slots*k*b_t)
        J['w_t', 'n_slots'] = -(2*pi*rot_or*B_g)/(n_slots**2*k*b_t)
        J['w_t', 'k']   = -(2*pi*rot_or*B_g)/(n_slots*k**2*b_t)
        J['w_t', 'b_t'] = -(2*pi*rot_or*B_g)/(n_slots*k*b_t**2)

        # Slot Depth
        J['s_d', 'radius_motor'] = 1 
        J['s_d', 'rot_or'] = ds_d__drot_or
        J['s_d', 'gap'] = -1
        J['s_d', 'B_g'] = ds_d__dB_g
        J['s_d', 'n_m'] = -dw_sy__dn_m
        J['s_d', 'k'] = ds_d__dk
        J['s_d', 'b_sy'] = -dw_sy__db_sy

        #Rotor Inner Radius
        J['rot_ir', 'rot_or'] = 1  - dw_ry__drot_or
        J['rot_ir', 't_mag'] = -1
        J['rot_ir', 'B_g'] = - dw_ry__dB_g 
        J['rot_ir', 'n_m'] = -dw_ry__dn_m
        J['rot_ir', 'k'] = - dw_ry__dk
        J['rot_ir', 'b_ry'] = -dw_ry__db_ry

        # Stator Inner Radius
        J['sta_ir', 'rot_or'] =  1
        J['sta_ir', 'gap'] =  1

        # Slot Area
        J['slot_area', 'n_slots'] = dslot_area__dn_slots
        J['slot_area', 'radius_motor'] = dslot_area__dradius_motor
        J['slot_area', 'rot_or'] = dslot_area__drot_or
        J['slot_area', 'B_g'] = dslot_area__dBg
        J['slot_area', 'n_m'] = dslot_area__dn_m
        J['slot_area', 'k'] = dslot_area__dk
        J['slot_area', 'b_sy'] = dslot_area__db_sy
        J['slot_area', 'gap'] = dslot_area__dgap
        J['slot_area', 'b_t'] = dslot_area__db_t

        # Slot Width 
        J['w_slot', 'n_slots'] = dslot_area__dn_slots/s_d 
        J['w_slot', 'radius_motor'] = dslot_area__dradius_motor/s_d - slot_area/s_d**2
        J['w_slot', 'rot_or'] = dslot_area__drot_or/s_d - slot_area/s_d**2 * ds_d__drot_or
        J['w_slot', 'B_g'] = dslot_area__dBg/s_d - slot_area/s_d**2 * ds_d__dB_g
        J['w_slot', 'n_m'] = dslot_area__dn_m/s_d - slot_area/s_d**2 * ds_d__dn_m
        J['w_slot', 'k'] = dslot_area__dk/s_d - slot_area/s_d**2 * ds_d__dk
        J['w_slot', 'b_sy'] = dslot_area__db_sy/s_d - slot_area/s_d**2 * ds_d__db_sy
        J['w_slot', 'gap'] = dslot_area__dgap/s_d + slot_area/s_d**2
        J['w_slot', 'b_t'] = dslot_area__db_t/s_d

        # Current density
        root8 = np.sqrt(8) # 2 * sqrt(2) = sqrt(4) * sqrt(2) = sqrt(8)
        J['J', 'n_turns'] =  root8*I/(1e6*k_wb*slot_area)
        J['J', 'I_required'] = root8*n_turns/(1e6*k_wb*slot_area)
        J['J', 'k_wb'] = -root8*I*n_turns/(1e6* k_wb**2 *slot_area)

        const_term = -root8*n_turns*I/(1e6*k_wb*slot_area**2)
        J['J', 'n_slots'] =  const_term * dslot_area__dn_slots
        J['J', 'radius_motor'] = const_term * dslot_area__dradius_motor    
        J['J', 'rot_or'] = const_term * dslot_area__drot_or
        J['J', 'B_g'] =  const_term * dslot_area__dBg
        J['J', 'n_m'] =  const_term * dslot_area__dn_m
        J['J', 'k'] =  const_term * dslot_area__dk
        J['J', 'b_sy'] =  const_term * dslot_area__db_sy
        J['J', 'gap'] =  const_term * dslot_area__dgap
        J['J', 'b_t'] =  const_term * dslot_area__db_t

class MotorMassComp(om.ExplicitComponent):

    def setup(self):
        self.add_input('rho', 8110.2, units='kg/m**3', desc='density of hiperco-50')
        self.add_input('radius_motor', .080, units='m', desc='motor outer radius')           
        self.add_input('n_slots', 20, desc='number of slots')                           
        self.add_input('sta_ir', .070, units='m', desc='stator inner radius')       
        self.add_input('w_t', .0045, units='m', desc='tooth width')                        
        self.add_input('stack_length', 0.0345, units='m', desc='length of stack')  
        self.add_input('s_d', 0.012, units='m', desc='slot depth')                         
        self.add_input('rot_or', 0.0615, units='m', desc='rotor outer radius')
        self.add_input('rot_ir', 0.0515, units='m', desc='rotor inner radius')
        self.add_input('t_mag', .0045, units='m', desc='magnet thickness')
        self.add_input('rho_mag', 7500, units='kg/m**3', desc='density of magnet')
        self.add_input('rho_wire', 8940, units='kg/m**3', desc='Density of wire: Cu=8940')
        self.add_input('L_wire', 10, units='m', desc='length of the wire')
        self.add_input('r_litz', .001, units='m', desc='radius of the wire')

        self.add_output('mag_mass', 0.5, units='kg', desc='mass of magnets')
        self.add_output('sta_mass', 1.0, units='kg', desc='mass of stator')
        self.add_output('rot_mass', 1.0, units='kg', desc='mass of rotor')
        self.add_output('wire_mass', 1.0, units='kg', desc='mass of the windings')
        self.add_output('motor_mass', 2.5, units='kg', desc='mass of motor')
       
        self.declare_partials('mag_mass', ['rot_or', 't_mag', 'rho_mag', 'stack_length'])
        self.declare_partials('rot_mass', ['rot_or', 't_mag', 'rot_ir', 'rho', 'stack_length'])
        self.declare_partials('sta_mass', ['rho', 'stack_length', 'radius_motor', 'sta_ir', 's_d', 'n_slots', 'w_t'])
        self.declare_partials('wire_mass', ['L_wire', 'rho_wire', 'r_litz'])
        self.declare_partials('motor_mass', ['rot_or', 't_mag', 'rho_mag', 'stack_length', 'rot_ir', 'rho', 'radius_motor',
                                             'sta_ir', 's_d', 'n_slots', 'w_t', 'L_wire', 'rho_wire', 'r_litz'])


    def compute(self,inputs,outputs):
        rho=inputs['rho']
        radius_motor=inputs['radius_motor']
        n_slots=inputs['n_slots']
        sta_ir=inputs['sta_ir']
        w_t=inputs['w_t']
        stack_length=inputs['stack_length']
        s_d=inputs['s_d']
        rot_ir=inputs['rot_ir']
        rot_or=inputs['rot_or']
        t_mag=inputs['t_mag']
        rho_mag=inputs['rho_mag']
        rho_wire=inputs['rho_wire']
        L_wire=inputs['L_wire']
        r_litz=inputs['r_litz']

        outputs['sta_mass']  = rho * stack_length * ((pi * radius_motor**2)-(pi * (sta_ir+s_d)**2)+(n_slots*(w_t*s_d)))
        outputs['rot_mass']  = (pi*(rot_or - t_mag)**2 - pi*rot_ir**2) * rho * stack_length
        outputs['mag_mass']  = (((pi*rot_or**2) - (pi*(rot_or-t_mag)**2))) * rho_mag * stack_length
        outputs['wire_mass'] = L_wire*rho_wire* (2*pi*r_litz**2 )
        outputs['motor_mass'] = outputs['sta_mass'] + outputs['rot_mass'] + outputs['mag_mass'] + outputs['wire_mass']

        # outputs['sta_cp'] = sta_mass * hiperco_cp / motor_mass
        # outputs['rot_cp'] = sta_mass * hiperco_cp / motor_mass
        # outputs['mag_cp'] = mag_mass * neo_cp / motor_mass
        # outputs['wire_cp'] = wire_mass * copper_cp / motor_mass


    def compute_partials(self,inputs,J):
        rho=inputs['rho']
        radius_motor=inputs['radius_motor']
        n_slots=inputs['n_slots']
        sta_ir=inputs['sta_ir']
        w_t=inputs['w_t']
        stack_length=inputs['stack_length']
        rho_mag=inputs['rho_mag']
        t_mag=inputs['t_mag']
        s_d=inputs['s_d']
        rot_ir=inputs['rot_ir']
        rot_or=inputs['rot_or']
        rho_wire=inputs['rho_wire']
        L_wire=inputs['L_wire']
        r_litz=inputs['r_litz']

        J['sta_mass', 'rho'] = stack_length * ((pi * radius_motor**2)-(pi * (sta_ir+s_d)**2)+(n_slots*(w_t*s_d)))
        J['sta_mass', 'stack_length'] = rho * ((pi * radius_motor**2)-(pi * (sta_ir+s_d)**2)+(n_slots*(w_t*s_d)))
        J['sta_mass', 'radius_motor'] = rho * stack_length * (pi * radius_motor)*2
        J['sta_mass', 'sta_ir'] = -2*pi*rho*stack_length*(sta_ir+s_d)
        J['sta_mass', 's_d'] = rho*stack_length*(-2*pi*(sta_ir+s_d) + n_slots*w_t)
        J['sta_mass', 'n_slots'] = rho*stack_length*w_t*s_d
        J['sta_mass', 'w_t'] = rho*stack_length*n_slots*s_d

        J['rot_mass', 'rho'] = (pi*(rot_or - t_mag)**2 - pi*rot_ir**2)  * stack_length
        J['rot_mass', 'rot_or'] = (2*pi*(rot_or - t_mag)) * rho * stack_length
        J['rot_mass', 't_mag'] =  (-2*pi*(rot_or - t_mag)) * rho * stack_length
        J['rot_mass', 'rot_ir'] = -2*pi*rot_ir * rho * stack_length
        J['rot_mass', 'stack_length'] = ((pi*(rot_or - t_mag)**2 - pi*rot_ir**2) * rho)

        J['mag_mass', 'rot_or'] = ((2*pi*rot_or - 2*pi*(rot_or-t_mag)))*stack_length*rho_mag
        J['mag_mass', 't_mag'] = 2 * pi * rho_mag * stack_length * (rot_or-t_mag)
        J['mag_mass', 'rho_mag'] = (((pi*rot_or**2) - (pi*(rot_or-t_mag)**2))) * stack_length
        J['mag_mass', 'stack_length'] = (((pi*rot_or**2) - (pi*(rot_or-t_mag)**2))) * rho_mag

        J['wire_mass', 'L_wire'] = rho_wire* (2*pi*r_litz**2 )
        J['wire_mass', 'rho_wire'] = L_wire* (2*pi*r_litz**2 )
        J['wire_mass', 'r_litz'] = L_wire*rho_wire* (4*pi*r_litz )

        J['motor_mass', 'rot_or'] = ((2*pi*rot_or - 2*pi*(rot_or-t_mag)))*stack_length*rho_mag
        J['motor_mass', 't_mag'] = (-2*pi*(rot_or - t_mag)) * rho * stack_length + 2 * pi * rho_mag * stack_length * (rot_or-t_mag)
        J['motor_mass', 'rho_mag'] = (((pi*rot_or**2) - (pi*(rot_or-t_mag)**2))) * stack_length
        J['motor_mass', 'stack_length'] = (rho * ((pi * radius_motor**2)-(pi * (sta_ir+s_d)**2)+(n_slots*(w_t*s_d)))) + ((pi*(rot_or - t_mag)**2 - pi*rot_ir**2) * rho) + ((((pi*rot_or**2) - (pi*(rot_or-t_mag)**2))) * rho_mag)
        J['motor_mass', 'rot_ir'] = -2*pi*rot_ir * rho * stack_length
        J['motor_mass', 'rho'] = (stack_length * ((pi * radius_motor**2)-(pi * (sta_ir+s_d)**2)+(n_slots*(w_t*s_d)))) + ((pi*(rot_or - t_mag)**2 - pi*rot_ir**2)  * stack_length)
        J['motor_mass', 'radius_motor'] = rho * stack_length * (pi * radius_motor)*2
        J['motor_mass', 'sta_ir'] = -2*pi*rho*stack_length*(sta_ir+s_d)
        J['motor_mass', 's_d'] = rho*stack_length*(-2*pi*(sta_ir+s_d) + n_slots*w_t)
        J['motor_mass', 'n_slots'] = rho*stack_length*w_t*s_d
        J['motor_mass', 'w_t'] = rho*stack_length*n_slots*s_d
        J['motor_mass', 'L_wire'] = rho_wire* (2*pi*r_litz**2 )
        J['motor_mass', 'rho_wire'] = L_wire* (2*pi*r_litz**2 )
        J['motor_mass', 'r_litz'] = L_wire*rho_wire* (4*pi*r_litz )

class SpecificHeatComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('copper_cp', 386, units='J/kg/C', desc='specific heat for copper')
        self.add_input('hiperco_cp', 410, units='J/kg/C', desc='specific heat for hiperco')
        self.add_input('neo_cp', 190, units='J/kg/C', desc='specific heat for neodymium')
        self.add_input('mag_mass', 0.5, units='kg', desc='mass of magnets')
        self.add_input('sta_mass', 1.0, units='kg', desc='mass of stator')
        self.add_input('rot_mass', 1.0, units='kg', desc='mass of rotor')
        self.add_input('wire_mass', 1.0, units='kg', desc='mass of the windings')
        self.add_input('motor_mass', 2.5, units='kg', desc='mass of motor')

        self.add_output('mag_cp', units='J/kg/C', desc='Specific heat for magnets')
        self.add_output('sta_cp', units='J/kg/C', desc='Specific heat for stator')
        self.add_output('rot_cp', units='J/kg/C', desc='Specific heat for rotor')
        self.add_output('wire_cp', units='J/kg/C', desc='Specific heat for wires')

        self.declare_partials('mag_cp', ('neo_cp', 'mag_mass', 'motor_mass'))
        self.declare_partials('sta_cp', ('hiperco_cp', 'sta_mass', 'motor_mass'))
        self.declare_partials('rot_cp', ('hiperco_cp', 'rot_mass', 'motor_mass'))
        self.declare_partials('wire_cp', ('copper_cp', 'wire_mass', 'motor_mass'))

    def compute(self, inputs, outputs):
        copper_cp = inputs['copper_cp']
        hiperco_cp = inputs['hiperco_cp']
        neo_cp = inputs['neo_cp']
        mag_mass = inputs['mag_mass']
        sta_mass = inputs['sta_mass']
        rot_mass = inputs['rot_mass']
        wire_mass = inputs['wire_mass'] 
        motor_mass = inputs['motor_mass']

        outputs['sta_cp'] = sta_mass * hiperco_cp / motor_mass
        outputs['rot_cp'] = rot_mass * hiperco_cp / motor_mass
        outputs['mag_cp'] = mag_mass * neo_cp / motor_mass
        outputs['wire_cp'] = wire_mass * copper_cp / motor_mass

    def compute_partials(self,inputs,J):
        copper_cp = inputs['copper_cp']
        hiperco_cp = inputs['hiperco_cp']
        neo_cp = inputs['neo_cp']
        mag_mass = inputs['mag_mass']
        sta_mass = inputs['sta_mass']
        rot_mass = inputs['rot_mass']
        wire_mass = inputs['wire_mass'] 
        motor_mass = inputs['motor_mass']

        J['sta_cp', 'sta_mass'] =   hiperco_cp / motor_mass
        J['sta_cp', 'hiperco_cp'] = sta_mass / motor_mass
        J['sta_cp', 'motor_mass'] = -sta_mass * hiperco_cp / motor_mass**2

        J['rot_cp', 'rot_mass'] =   hiperco_cp / motor_mass
        J['rot_cp', 'hiperco_cp'] = rot_mass / motor_mass
        J['rot_cp', 'motor_mass'] = -rot_mass * hiperco_cp / motor_mass**2

        J['mag_cp', 'mag_mass'] =   neo_cp / motor_mass
        J['mag_cp', 'neo_cp'] =      mag_mass / motor_mass
        J['mag_cp', 'motor_mass'] = -mag_mass * neo_cp / motor_mass**2

        J['wire_cp', 'wire_mass'] =  copper_cp / motor_mass
        J['wire_cp', 'copper_cp'] =  wire_mass / motor_mass
        J['wire_cp', 'motor_mass'] = -wire_mass * copper_cp / motor_mass**2


if __name__ == "__main__":
    from openmdao.api import Problem

    nn = 1
    prob = Problem()
    # des_vars = prob.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=['*'])
    # des_vars.add_output('r_strand', np.ones(nn), units='m')
    # des_vars.add_output('mu_o', np.ones(nn), units='H/m')
    # des_vars.add_output('resistivity_wire', np.ones(nn), units='ohm*m')

    prob.model.add_subsystem('sys', MotorSizeComp(num_nodes=nn), promotes=['*'])

    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(method='cs', compact_print=True)

