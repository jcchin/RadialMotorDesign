import numpy as np
from math import pi

import openmdao.api as om

class MotorSizeComp(om.ExplicitComponent):

    def setup(self):
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
        self.add_input('n_turns', 12, desc='number of wire turns')
        self.add_input('I', 30, units='A', desc='RMS current')  # Imax
        self.add_input('k_wb', 0.65, desc='bare wire slot fill factor')

        self.add_output('w_ry', .004, units='m', desc='width of stator yoke')
        self.add_output('w_sy', .005, units='m', desc='width of stator yoke')
        self.add_output('w_t', 0.0048, units='m', desc='width of tooth')   
        self.add_output('s_d', .012, units='m', desc='slot depth')
        self.add_output('rot_ir', .061, units='m', desc='rotor inner radius')
        self.add_output('sta_ir', .070, units='m', desc='stator inner radius')
        self.add_output('slot_area', 0.0002, units='m**2', desc='area of one slot')
        self.add_output('w_slot', .015, units='m', desc='width of a slot')
        self.add_output('J', units='A/mm**2', desc='Current density')

        self.declare_partials('*','*', method='fd')
        # self.declare_partials('w_ry', ['rot_or', 'B_g', 'n_m', 'k', 'b_ry'])
        # self.declare_partials('w_sy', ['rot_or', 'B_g', 'n_m', 'k', 'b_sy'])
        # self.declare_partials('w_t', ['rot_or','B_g','n_slots','k','b_t'])
        # self.declare_partials('s_d', ['radius_motor', 'rot_or', 'gap', 'B_g', 'n_m', 'k', 'b_sy'])
        # self.declare_partials('rot_ir', ['rot_or', 't_mag', 'B_g', 'n_m', 'k', 'b_ry'])
        # self.declare_partials('sta_ir', ['rot_or', 'gap'])
        # self.declare_partials('slot_area', ['n_slots', 'radius_motor', 'rot_or', 'B_g', 'n_m', 'k', 'b_sy', 'gap', 'b_t'])
        # self.declare_partials('w_slot', ['n_slots', 'radius_motor', 'rot_or', 'B_g', 'n_m', 'k', 'b_sy', 'gap', 'b_t'])
        # self.declare_partials('J', ['n_turns', 'I', 'k_wb', 'n_slots', 'radius_motor', 'rot_or', 'B_g', 'n_m', 'k', 'b_sy', 'gap', 'b_t'])

    def compute(self,inputs,outputs):
        radius_motor = inputs['radius_motor']  # .0765
        gap = inputs['gap']
        B_g = inputs['B_g']
        n_m = inputs['n_m']
        k = inputs['k']
        b_ry = inputs['b_ry']
        t_mag = inputs['t_mag']
        n_turns = inputs['n_turns']
        I = inputs['I']
        k_wb = inputs['k_wb']
        b_sy= inputs['b_sy']
        n_slots = inputs['n_slots']
        b_t = inputs['b_t']
        rot_or = inputs['rot_or']
        # variable for pi*rot_or*...
        # replace eqns with outputs

        outputs['w_ry'] = (pi*rot_or*B_g)/(n_m*k*b_ry) 
        outputs['w_sy'] = (pi*rot_or*B_g)/(n_m*k*b_sy)
        outputs['w_t'] = (2*pi*rot_or*B_g) / (n_slots*k*b_t) 
        outputs['s_d'] = radius_motor - rot_or - gap - ((pi*rot_or*B_g)/(n_m*k*b_sy))
        outputs['rot_ir'] = (rot_or- t_mag) - outputs['w_ry'] 
        outputs['sta_ir'] = rot_or + gap
        outputs['slot_area'] = pi/n_slots*((radius_motor**2 - (2*radius_motor*pi*rot_or*B_g)/(n_m*k*b_sy) + ((pi*rot_or*B_g)/(n_m*k*b_sy))**2) -  (rot_or**2 + 2*rot_or*gap + gap**2 ))   - ( 1.05*(2*pi*rot_or*B_g)/(n_slots*k*b_t) * (radius_motor - rot_or - gap - (pi*rot_or*B_g)/(n_m*k*b_sy)) )
        outputs['w_slot']    = ( pi/n_slots*((radius_motor**2 - (2*radius_motor*pi*rot_or*B_g)/(n_m*k*b_sy) + ((pi*rot_or*B_g)/(n_m*k*b_sy))**2) -  (rot_or**2 + 2*rot_or*gap + gap**2 ))   - ( (2*pi*rot_or*B_g)/(n_slots*k*b_t) * (radius_motor - rot_or - gap - (pi*rot_or*B_g)/(n_m*k*b_sy)) ) ) / (radius_motor - rot_or - gap - ((pi*rot_or*B_g)/(n_m*k*b_sy)))         #outputs['slot_area'] / outputs['s_d']
        outputs['J'] = 2*n_turns*I*(2.**0.5)/(k_wb*(outputs['slot_area'])*1E6)

    # def compute_partials(self, inputs, J):

    #     radius_motor = inputs['radius_motor']
    #     gap = inputs['gap']
    #     B_g = inputs['B_g']
    #     n_m = inputs['n_m']
    #     k = inputs['k']
    #     b_ry = inputs['b_ry']
    #     t_mag = inputs['t_mag']
    #     n_turns = inputs['n_turns']
    #     I = inputs['I']
    #     k_wb = inputs['k_wb']
    #     b_sy= inputs['b_sy']
    #     n_slots = inputs['n_slots']
    #     b_t = inputs['b_t']
    #     rot_or = inputs['rot_or']

    #     # Rotor Yoke Width
    #     J['w_ry', 'rot_or'] = (pi*B_g)/(n_m*k*b_ry)
    #     J['w_ry', 'B_g'] = (pi*rot_or)/(n_m*k*b_ry)
    #     J['w_ry', 'n_m'] = -(pi*rot_or*B_g)/(n_m**2*k*b_ry)
    #     J['w_ry', 'k']   = -(pi*rot_or*B_g)/(n_m*k**2*b_ry)
    #     J['w_ry', 'b_ry'] = -(pi*rot_or*B_g)/(n_m*k*b_ry**2)

    #     # Stator Yoke Width
    #     J['w_sy', 'rot_or'] = (pi*B_g)/(n_m*k*b_sy)
    #     J['w_sy', 'B_g'] = (pi*rot_or)/(n_m*k*b_sy)
    #     J['w_sy', 'n_m'] = -(pi*rot_or*B_g)/(n_m**2*k*b_sy)
    #     J['w_sy', 'k']   = -(pi*rot_or*B_g)/(n_m*k**2*b_sy)
    #     J['w_sy', 'b_sy'] = -(pi*rot_or*B_g)/(n_m*k*b_sy**2)

    #     # Tooth Width
    #     J['w_t', 'rot_or'] = (2*pi*B_g)/(n_slots*k*b_t)
    #     J['w_t', 'B_g'] = (2*pi*rot_or)/(n_slots*k*b_t)
    #     J['w_t', 'n_slots'] = -(2*pi*rot_or*B_g)/(n_slots**2*k*b_t)
    #     J['w_t', 'k']   = -(2*pi*rot_or*B_g)/(n_slots*k**2*b_t)
    #     J['w_t', 'b_t'] = -(2*pi*rot_or*B_g)/(n_slots*k*b_t**2)

    #     # Slot Depth
    #     J['s_d', 'radius_motor'] = 1 - rot_or - gap - (pi*rot_or*B_g)/(n_m*k*b_sy)
    #     J['s_d', 'rot_or'] = radius_motor - 1 - gap - (pi*B_g)/(n_m*k*b_sy)
    #     J['s_d', 'gap'] = radius_motor - rot_or - 1 - (pi*rot_or*B_g)/(n_m*k*b_sy)
    #     J['s_d', 'B_g'] = radius_motor - rot_or - gap - (pi*rot_or)/(n_m*k*b_sy)
    #     J['s_d', 'n_m'] = radius_motor - rot_or - gap + (pi*rot_or*B_g)/(n_m**2*k*b_sy)
    #     J['s_d', 'k'] = radius_motor - rot_or - gap + (pi*rot_or*B_g)/(n_m*k**2*b_sy)
    #     J['s_d', 'b_sy'] = radius_motor - rot_or - gap + (pi*rot_or*B_g)/(n_m*k*b_sy**2)

    #     #Rotor Inner Radius
    #     J['rot_ir', 'rot_or'] = (1 - t_mag) - (pi*B_g)/(n_m*k*b_ry) 
    #     J['rot_ir', 't_mag'] = (rot_or- 1) - (pi*rot_or*B_g)/(n_m*k*b_ry)   
    #     J['rot_ir', 'B_g'] = (rot_or- t_mag) - (pi*rot_or)/(n_m*k*b_ry) 
    #     J['rot_ir', 'n_m'] = radius_motor - rot_or - gap + (pi*rot_or*B_g)/(n_m**2*k*b_ry)
    #     J['rot_ir', 'k'] = radius_motor - rot_or - gap + (pi*rot_or*B_g)/(n_m*k**2*b_ry)
    #     J['rot_ir', 'b_ry'] = radius_motor - rot_or - gap + (pi*rot_or*B_g)/(n_m*k*b_ry**2)

    #     # Stator Inner Radius
    #     J['sta_ir', 'rot_or'] =  1 + gap
    #     J['sta_ir', 'gap'] =  rot_or + 1

    #     # Slot Area
        J['slot_area', 'n_slots'] = -(pi/n_slots**2)*((radius_motor**2 - (2*radius_motor*pi*rot_or*B_g)/(n_m*k*b_sy) + ((pi*rot_or*B_g)/(n_m*k*b_sy))**2) -  \
                                    (rot_or**2 + 2*rot_or*gap + gap**2 )) - ((-2*pi*rot_or*B_g)/(n_slots**2 * k*b_t) * (radius_motor - rot_or - gap - (pi*rot_or*B_g)/(n_m*k*b_sy))) 
    #     J['slot_area', 'radius_motor'] = pi/n_slots*(2*radius_motor - (2*pi*rot_or*B_g)/(n_m*k*b_sy)) -  (2*pi*rot_or*B_g)/(n_slots*k*b_t)
    #     J['slot_area', 'rot_or'] = (pi/n_slots) * ((-2*radius_motor*pi*B_g)/(n_m*k*b_sy) + 2*rot_or*pi*B_g/(n_m*k*b_sy) - (2*rot_or + 2*gap)) - (2*pi*B_g/(n_slots*k*b_t) * (radius_motor - 2*rot_or - (2*pi*rot_or*B_g/(n_m*k*b_sy))))
    #     J['slot_area', 'B_g'] = (pi/n_slots) * (2*radius_motor*pi*rot_or/(n_m*k*b_sy) + 2*B_g*(pi*rot_or/n_m/k/b_sy)**2  ) - (2*pi*rot_or/n_slots/k/b_t * (radius_motor - rot_or - gap - (2*pi*rot_or*B_g/n_m/k/b_sy)))
    #     J['slot_area', 'n_m'] = (pi/n_slots) * ((2*radius_motor*pi*rot_or*B_g)/(n_m**2 *k*b_sy) - ((pi*rot_or*B_g)/(k*b_sy))**2 * n_m**-3) - ((2*pi**2*rot_or**2*B_g**2)/(n_slots*k**2*b_t*b_sy*n_m**2))
    #     J['slot_area', 'k'] =   (pi/n_slots) * ((2*radius_motor*pi*rot_or*B_g)/(n_m*k**2 *b_sy) - ((pi*rot_or*B_g)/(n_m*b_sy))**2 * k**-3) - ((-2*pi*rot_or*B_g)/(n_slots*k**2 *b_t) * (radius_motor - rot_or - gap) - (4*pi**2 *rot_or**2 *B_g**2)/(n_m* k**3 *b_sy*b_t*n_slots))
    #     J['slot_area', 'b_sy'] =  (pi/n_slots) * ((2*radius_motor*pi*rot_or*B_g)/(n_m*k *b_sy**2) - ((pi*rot_or*B_g)/(n_m*k))**2 * b_sy**-3) - ((2*pi*rot_or*B_g)/(n_slots*k*b_t) * ((pi*rot_or*B_g)/(n_m*k*b_sy**2)))
    #     J['slot_area', 'gap'] = pi/n_slots*(-2*rot_or + 2*gap) - ((-2*pi*rot_or*B_g)/(n_slots*k*b_t))
    #     J['slot_area', 'b_t'] = - ( (-2*pi*rot_or*B_g)/(n_slots*k *b_t**2) * (radius_motor - rot_or - gap) - (2*pi**2 *rot_or**2 *B_g**2)/(n_m* k**2 *b_sy*b_t**2 *n_slots))

    #     # Slot Width 
    #     J['w_slot', 'n_slots'] = ((4*pi**2*B_g*rot_or)/(k*n_slots**3*b_t)) - ((pi*((pi**2*B_g**2*rot_or**2)/(k**2*n_m**2*b_sy**2) - (2*pi*B_g*radius_motor*rot_or)/(k*n_m*b_sy) + gap**2- 2*gap*rot_or + radius_motor**2 - rot_or)) / (n_slots**2 * ((-pi*B_g*rot_or)/(k*n_m*b_sy) - gap + radius_motor - rot_or))  )
    #     J['w_slot', 'radius_motor'] = (( pi*(2*radius_motor - (2*pi*B_g*rot_or)/(k*n_m*b_sy)) )/( n_slots* ((-pi*B_g*rot_or)/(k*n_m*b_sy) - gap + radius_motor - rot_or))) - (pi*((pi**2*B_g**2*rot_or**2)/(k**2*n_m**2*b_sy**2) - (2*pi*B_g*radius_motor*rot_or)/(k*n_m*b_sy) + gap**2-2*gap*rot_or*radius_motor**2 - rot_or )/( n_slots *( (-pi*B_g*rot_or)/(k*n_m*b_sy) - gap + radius_motor - rot_or )**2 )  )
    #     J['w_slot', 'rot_or'] = ((-pi*(-pi*B_g/(k*n_m*b_sy)-1) * ((pi**2*B_g**2*rot_or**2)/(k**2*n_m**2*b_sy**2) - (2*pi*B_g*radius_motor*rot_or)/(k*n_m*b_sy) + gap**2-2*gap*rot_or+radius_motor**2- rot_or)) / (n_slots*( (-pi*B_g*rot_or)/(k*n_m*b_sy)-gap+radius_motor- rot_or )**2)) + (pi*((2*pi**2*B_g*rot_or)/(k**2*n_m**2*b_sy**2) - ((2*pi*B_g*radius_motor)/(k*n_m*b_sy) -2*gap- n_slots ) ) / (  (n_slots*((-pi*B_g*rot_or)/(k*n_m*b_sy)-gap+radius_motor- rot_or) ))) - (2*pi**2*B_g)/(k*n_slots**2*b_t)  
    #     J['w_slot', 'B_g'] = (pi**2*rot_or*(((pi*B_g*rot_or)/(k*n_m*b_sy))**2 - (2*pi*B_g*radius_motor*rot_or)/(n_m*k*b_sy) + gap**2 - 2*gap*rot_or + radius_motor**2 - rot_or)) / (k*n_slots*n_m*b_sy*((-pi*B_g*rot_or)/(k*n_m*b_sy) -gap+radius_motor-rot_or )**2)   +   ((2*pi*B_g*( ((pi*rot_or)/(k*n_m*b_sy))**2 - (2*pi*radius_motor*rot_or)/(B_g*k*n_m*b_sy)))/( n_slots*((-pi*B_g*rot_or)/(k*n_m*b_sy) - gap+radius_motor-rot_or )))   -   ((2*pi**2*rot_or)/(k*n_slots**2*b_t))
    #     # J['w_slot', 'n_m'] = ((2*pi**2*/n_m*((B_g*radius_motor*rot_or)/(k*n_m*b_sy) - pi*((B_g*rot_or)/(k*n_m*b_sy))**2)) / n_slots*((-pi*B_g*rot_or)/(k*n_m*b_sy)-gap+radius_motor-rot_or)) - ((pi**2*B_g*rot_or*(((pi*B_g*rot_or)/(k*n_m*b_sy))**2 - (2*pi*B_g*radius_motor*rot_or)/(k*n_m*b_sy) + gap**2-2*gap*rot_or + radius_motor**2 - rot_or)) / (k*n_slots*n_m**2*b_sy*((-pi*B_g*rot_or)/(k*n_m*b_sy) - gap + radius_motor - rot_or)**2))
    #     J['w_slot', 'k'] = -(pi**2*B_g*rot_or*( ((pi*B_g*rot_or)/(k*n_m*b_sy))**2 - (2*radius_motor*pi*B_g*rot_or)/(k*n_m*b_sy) + gap**2 - 2*gap*rot_or + radius_motor**2 - rot_or)) / (k**2*n_slots*n_m*b_sy*((-pi*B_g*rot_or)/(k*n_m*b_sy) -gap + radius_motor - rot_or)**2)  +  ((pi*((2*radius_motor*pi*B_g*rot_or)/(k**2*n_m*b_sy) - (2*pi**2*B_g**2*rot_or**2)/(k**3*n_m**2*b_sy**2))) / (n_slots*((-pi*B_g*rot_or)/(k*n_m*b_sy) - gap+radius_motor-rot_or))) + ((2*pi**2*B_g*rot_or)/(k**2*n_slots**2*b_t))
    #     J['w_slot', 'b_sy'] = ((pi*((2*pi*B_g*radius_motor*rot_or)/(k*n_m*b_sy) - (2*pi**2*B_g**2*rot_or**2)/(k**2*n_m**2*b_sy**3))) / (n_slots*(  (-pi*B_g*rot_or)/(k*n_m*b_sy) - gap + radius_motor - rot_or  ))) - (  pi**2*B_g*rot_or*(  ((pi*B_g*rot_or)/(k*n_m*b_sy))**2 - (2*pi*B_g*radius_motor*rot_or)/(k*n_m*b_sy) + gap**2- 2*gap*rot_or+ radius_motor**2- rot_or  ) / ( k*n_slots*n_m*b_sy**2*( (-pi*B_g*rot_or)/(k*n_m*b_sy)-gap+radius_motor-rot_or )**2 )  )
    #     J['w_slot', 'gap'] = ((pi*(  ((pi*B_g*rot_or)/(k*n_m*b_sy))**2 - (2*pi*B_g*rot_or*radius_motor)/(k*n_m*b_sy) + gap**2- 2*gap*rot_or+ radius_motor**2- rot_or  )) / (n_slots*(  (pi*B_g*rot_or)/(k*n_m*b_sy) - gap+ radius_motor- rot_or  )**2)) + (pi*(2*gap- 2*rot_or))/(n_slots*( (-pi*B_g*rot_or)/(k*n_m*b_sy) - gap+ radius_motor- rot_or))
    #     J['w_slot', 'b_t'] = (2*pi**2*B_g*rot_or)/(k*n_slots**2*b_t**2)

    #     # Current density
    #     J['J', 'n_turns'] = 2*I*(2.**0.5)/(k_wb*(pi/n_slots*((radius_motor**2 - (2*radius_motor*pi*rot_or*B_g)/(n_m*k*b_sy) + ((pi*rot_or*B_g)/(n_m*k*b_sy))**2) -  (rot_or**2 + 2*rot_or*gap + gap**2 ))   - ( 1.05*(2*pi*rot_or*B_g)/(n_slots*k*b_t) * (radius_motor - rot_or - gap - (pi*rot_or*B_g)/(n_m*k*b_sy)) ))*1E6)
    #     J['J', 'I'] = 2*n_turns*(2.**0.5)/(k_wb*(pi/n_slots*((radius_motor**2 - (2*radius_motor*pi*rot_or*B_g)/(n_m*k*b_sy) + ((pi*rot_or*B_g)/(n_m*k*b_sy))**2) -  (rot_or**2 + 2*rot_or*gap + gap**2 ))   - ( 1.05*(2*pi*rot_or*B_g)/(n_slots*k*b_t) * (radius_motor - rot_or - gap - (pi*rot_or*B_g)/(n_m*k*b_sy)) ))*1E6)
    #     J['J', 'k_wb'] = -2*n_turns*I*(2.**0.5)/(k_wb**2*(pi/n_slots*((radius_motor**2 - (2*radius_motor*pi*rot_or*B_g)/(n_m*k*b_sy) + ((pi*rot_or*B_g)/(n_m*k*b_sy))**2) -  (rot_or**2 + 2*rot_or*gap + gap**2 ))   - ( 1.05*(2*pi*rot_or*B_g)/(n_slots*k*b_t) * (radius_motor - rot_or - gap - (pi*rot_or*B_g)/(n_m*k*b_sy)) ))*1E6)
    #     # J['J', 'n_slots'] = (2*sqrt{2}*I*k**2*n_m*b_t*n_turns*b_sy)/(rot_or*(pi*B_g**2*(pi*k*b_t + 6.59734*rot_or) + 2*B_g*k*(n*b_sy*(3.29867*g + 3.29867*rot_or) - radius_motor*(3.29867*n_m*b_sy + pi**2*b_t))) + pi*k**2*n_m*b_t*b_sy*(gap**2 - 2*gap*rot_or + m**2 - rot_or**2))
    #     J['J', 'radius_motor'] = (5.65685*(3.29867*I*B_g*k**3*n_slots*n_m**2*rot_or*b_t*n_turns*b_sy**2 + 9.8696*I*B_g*k**3*n_slots*n_m*rot_or*b_t**2*n_turns*b_sy - pi*I*k**4*n_slots*radius_motor*n_m**2*b_t**2*n_turns*b_sy**2))/(9.8696*B_g**2*k*rot_or*b_t + 20.7262*B_g**2*rot_or**2 + 6.59734*B_g*gap*k*n_m*rot_or*b_sy - 6.59734*B_g*k*radius_motor*n_m*rot_or*b_sy - 19.7392*B_g*k*radius_motor*rot_or*b_t + 6.59734*B_g*k*n_m*rot_or**2*b_sy + pi*gap**2*k**2*n_m*b_t*b_sy - 6.28319*gap*k**2*n_m*rot_or*b_t*b_sy + pi*k**2*radius_motor**2*n_m*b_t*b_sy - pi*k**2*n_m*rot_or**2*b_t*b_sy)**2
    #     J['J', 'rot_or'] = (1.80063*I*k**2*n_slots*n_m*b_t*n_turns*b_sy*(B_g**2*(-1.5708*k*b_t - 6.59734*rot_or) + B_g*k*(-1.05*gap*n_m*b_sy + 1.05*radius_motor*n_m*b_sy + pi*radius_motor*b_t - 2.1*n_m*rot_or*b_sy) + k**2*n_m*b_t*b_sy*(gap + rot_or)))/(B_g**2*rot_or*(-pi*k*b_t - 6.59734*rot_or) + B_g*k*rot_or*(-2.1*gap*n_m*b_sy + 2.1*radius_motor*n_m*b_sy + 6.28319*radius_motor*b_t - 2.1*n_m*rot_or*b_sy) + k**2*n_m*b_t*b_sy*(-gap**2 + 2*gap*rot_or - radius_motor**2 + rot_or**2))**2
    #     J['J', 'B_g'] = -(1.89066*I*k**2*n_slots*n_m*rot_or*b_t*n_turns*b_sy*(B_g*(2.99199*k*b_t + 6.28319*rot_or) + k*(gap*n_m*b_sy - radius_motor*n_m*b_sy - 2.99199*radius_motor*b_t + n_m*rot_or*b_sy)))/(B_g**2*rot_or*(-pi*k*b_t - 6.59734*rot_or) + B_g*k*rot_or*(-2.1*gap*n_m*b_sy + 2.1*radius_motor*n_m*b_sy + 6.28319*radius_motor*b_t - 2.1*n_m*rot_or*b_sy) + k**2*n_m*b_t*b_sy*(-gap**2 + 2*gap*rot_or - radius_motor**2 + rot_or**2))**2
    #     J['J', 'n_m'] = (2.82843*k**2*n_slots*b_t*b_sy*(9.8696*I*B_g**2*k*rot_or*b_t*n_turns + 20.7262*I*B_g**2*rot_or**2*n_turns - 19.7392*I*B_g*k*radius_motor*rot_or*b_t*n_turns))/(9.8696*B_g**2*k*rot_or*b_t + 20.7262*B_g**2*rot_or**2 + 6.59734*B_g*gap*k*n_m*rot_or*b_sy - 6.59734*B_g*k*radius_motor*n_m*rot_or*b_sy - 19.7392*B_g*k*radius_motor*rot_or*b_t + 6.59734*B_g*k*n_m*rot_or**2*b_sy + pi*gap**2*k**2*n_m*b_t*b_sy - 6.28319*gap*k**2*n_m*rot_or*b_t*b_sy + pi*k**2*radius_motor**2*n_m*b_t*b_sy - pi*k**2*n_m*rot_or**2*b_t*b_sy)**2
    #     J['J', 'k'] = (2.82843*(9.8696*I*B_g**2*k**2*n_slots*n_m*rot_or*b_t**2*n_turns*b_sy + 41.4523*I*B_g**2*k*n_slots*n_m*rot_or**2*b_t*n_turns*b_sy + 6.59734*I*B_g*gap*k**2*n_slots*n_m**2*rot_or*b_t*n_turns*b_sy**2 - 6.59734*I*B_g*k**2*n_slots*radius_motor*n_m**2*rot_or*b_t*n_turns*b_sy**2 - 19.7392*I*B_g*k**2*n_slots*radius_motor*n_m*rot_or*b_t**2*n_turns*b_sy + 6.59734*I*B_g*k**2*n_slots*n_m**2*rot_or**2*b_t*n_turns*b_sy**2))/(9.8696*B_g**2*k*rot_or*b_t + 20.7262*B_g**2*rot_or**2 + 6.59734*B_g*gap*k*n_m*rot_or*b_sy - 6.59734*B_g*k*radius_motor*n_m*rot_or*b_sy - 19.7392*B_g*k*radius_motor*rot_or*b_t + 6.59734*B_g*k*n_m*rot_or**2*b_sy + pi*gap**2*k**2*n_m*b_t*b_sy - 6.28319*gap*k**2*n_m*rot_or*b_t*b_sy + pi*k**2*radius_motor**2*n_m*b_t*b_sy - pi*k**2*n_m*rot_or**2*b_t*b_sy)**2
    #     J['J', 'b_sy'] = (2.82843*k**2*n_slots*n_m*b_t*(9.8696*I*B_g**2*k*rot_or*b_t*n_turns + 20.7262*I*B_g**2*rot_or**2*n_turns - 19.7392*I*B_g*k*radius_motor*rot_or*b_t*n_turns))/(9.8696*B_g**2*k*rot_or*b_t + 20.7262*B_g**2*rot_or**2 + 6.59734*B_g*gap*k*n_m*rot_or*b_sy - 6.59734*B_g*k*radius_motor*n_m*rot_or*b_sy - 19.7392*B_g*k*radius_motor*rot_or*b_t + 6.59734*B_g*k*n_m*rot_or**2*b_sy + pi*gap**2*k**2*n_m*b_t*b_sy - 6.28319*gap*k**2*n_m*rot_or*b_t*b_sy + pi*k**2*radius_motor**2*n_m*b_t*b_sy - pi*k**2*n_m*rot_or**2*b_t*b_sy)**2
    #     J['J', 'gap'] = (1.80063*I*k**3*n_slots*n_m**2*b_t*n_turns*b_sy**2*(k*b_t*(rot_or - gap) - 1.05*B_g*rot_or))/(B_g**2*rot_or*(-pi*k*b_t - 6.59734*rot_or) + B_g*k*rot_or*(-2.1*gap*n_m*b_sy + 2.1*radius_motor*n_m*b_sy + 6.28319*radius_motor*b_t - 2.1*n_m*rot_or*b_sy) + k**2*n_m*b_t*b_sy*(-gap**2 + 2*gap*rot_or - radius_motor**2 + rot_or**2))**2
    #     J['J', 'b_t'] = (I*B_g*k**2*n_slots*n_m*rot_or*n_turns*b_sy*(58.6225*B_g*rot_or + k*n_m*b_sy*(18.6601*gap - 18.6601*radius_motor + 18.6601*rot_or)))/(k*n_m*b_sy*(B_g*rot_or*(6.59734*gap - 6.59734*radius_motor + 6.59734*rot_or) + k*b_t*(pi*gap**2 - 6.28319*gap*rot_or + pi*radius_motor**2 - pi*rot_or**2)) + B_g*rot_or*(9.8696*B_g*k*b_t + 20.7262*B_g*rot_or - 19.7392*k*radius_motor*b_t))**2


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

        self.add_output('mag_mass', 0.5, units='kg', desc='mass of magnets')
        self.add_output('sta_mass', 1.0, units='kg', desc='mass of stator')
        self.add_output('rot_mass', 1.0, units='kg', desc='weight of rotor')
        # self.add_output('motor_mass', 2.0, units='kg', desc='total mass of motor')
        
        self.declare_partials('*','*', method='fd')

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

        outputs['sta_mass'] = rho * stack_length * ((pi * radius_motor**2)-(pi * (sta_ir+s_d)**2)+(n_slots*(w_t*s_d)))
        outputs['rot_mass'] = (pi*(rot_or - t_mag)**2 - pi*rot_ir**2) * rho * stack_length
        outputs['mag_mass'] = (((pi*rot_or**2) - (pi*(rot_or-t_mag)**2))) * rho_mag * stack_length
        # outputs['motor_mass'] = outputs['sta_mass'] + outputs['rot_mass'] + outputs['mag_mass']

    def compute_partials(self,inputs,J):
        rho=inputs['rho']
        radius_motor=inputs['radius_motor']
        n_slots=inputs['n_slots']
        sta_ir=inputs['sta_ir']
        w_t=inputs['w_t']
        stack_length=inputs['stack_length']

        J['sta_mass', 'rho'] = stack_length * ((pi * radius_motor**2)-(pi * (sta_ir+s_d)**2)+(n_slots*(w_t*s_d)))
        J['sta_mass', 'stack_length'] = rho * ((pi * radius_motor**2)-(pi * (sta_ir+s_d)**2)+(n_slots*(w_t*s_d)))
        J['sta_mass', 'radius_motor'] = 2*rho * stack_length * (pi * radius_motor**3)
        J['sta_mass', 'sta_ir'] = rho * stack_length * ((-(pi * (sta_ir+s_d)**2)+(n_slots*(w_t*s_d)))
        J['sta_mass', 's_d'] = 
        J['sta_mass', 'n_slots'] = 
        J['sta_mass', 'w_t'] = 

        J['rot_mass', 'rho'] = 
        J['rot_mass', 't_mag'] = 
        J['rot_mass', 'rot_ir'] = 
        J['rot_mass', 'stack_length'] = 

        J['mag_mass', 'rot_or'] = 
        J['mag_mass', 't_mag'] = 
        J['mag_mass', 'rho_mag'] = 
        J['mag_mass', 'stack_length'] = 

        

# ---------------------------------------      
 
