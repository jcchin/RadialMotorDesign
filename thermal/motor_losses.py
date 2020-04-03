# Future: Can include litz wire design from "Simplified Desigh Method for Litz Wire" C.R. Sullivan, R.Y. Zhang

from __future__ import absolute_import
import numpy as np
from math import pi

import openmdao.api as om


class WindingLossComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
    

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('rpm', 4000*np.ones(nn), units='rpm', desc='Rotation speed')
        self.add_input('n_m', 20, desc='Number of magnets')
        self.add_input('mu_o', 1.2566e-6, units='H/m', desc='permeability of free space')    
        self.add_input('mu_r', 1.0, units='H/m', desc='relative magnetic permeability of ferromagnetic materials') 
        self.add_input('r_strand', 0.0001605, units='m', desc='radius of one strand of litz wire')
        self.add_input('T_windings', 150, units='C', desc='operating temperature of windings')
        self.add_input('T_coeff_cu', 0.00393, desc='temperature coefficient for copper')
        self.add_input('resistivity_wire', 1.724e-8, units='ohm*m', desc='resisitivity of Cu at 20 degC')
        self.add_input('I', 30*np.ones(nn), units='A', desc='RMS current into motor')
        self.add_input('stack_length', 0.035, units='m', desc='axial length of stator')
        self.add_input('n_slots', 24, desc='number of slots')
        self.add_input('n_turns', 12, desc='number of winding turns')
        self.add_input('n_strands', 41, desc='number of strands in litz wire')        
        self.add_input('AC_power_factor', 0.5*np.ones(nn), desc='litz wire AC power factor')

        self.add_output('f_e', 900*np.ones(nn), units = 'Hz', desc='electrical frequency')
        self.add_output('r_litz', 0.0011, units='m', desc='radius of whole litz wire')
        self.add_output('L_wire', 10, units='m', desc='length of wire for one phase')
        self.add_output('temp_resistivity', 1.724e-8, units='ohm*m', desc='temp dependent resistivity')
        self.add_output('R_dc', 1, units='ohm', desc= 'DC resistance')
        self.add_output('skin_depth', 0.001*np.ones(nn), units='m', desc='skin depth of wire')
        self.add_output('A_cu', .005, units='m**2', desc='total area of copper in one slot')
        self.add_output('P_dc', 277*np.ones(nn), units='W ', desc= 'Power loss from dc resistance')
        self.add_output('P_ac', 100*np.ones(nn), units='W ', desc= 'Power loss from ac resistance')
        self.add_output('P_wire', 400*np.ones(nn), units='W ', desc= 'total power loss from wire')

        r = c = np.arange(nn)  # for scalar variables only

        self.declare_partials('f_e', ['n_m', 'rpm'], rows=r, cols=c)
        self.declare_partials('r_litz', ['n_strands', 'r_strand'])
        self.declare_partials('L_wire', ['n_slots', 'n_turns', 'stack_length'])
        self.declare_partials('temp_resistivity', ['resistivity_wire', 'T_coeff_cu', 'T_windings'])
        self.declare_partials('R_dc', ['resistivity_wire', 'T_coeff_cu', 'T_windings', 'n_slots', 'n_turns', 'stack_length', 'r_strand'])
        self.declare_partials('skin_depth', ['resistivity_wire', 'T_coeff_cu', 'T_windings', 'n_m', 'rpm', 'mu_r', 'mu_o'], rows=r, cols=c)
        self.declare_partials('A_cu', ['n_turns', 'n_strands', 'r_strand'])
        self.declare_partials('R_dc', ['resistivity_wire', 'T_coeff_cu', 'T_windings', 'n_slots', 'n_turns', 'stack_length', 'r_strand'])
        self.declare_partials('skin_depth', ['resistivity_wire', 'T_coeff_cu', 'T_windings', 'n_m', 'rpm', 'mu_r', 'mu_o'], rows=r, cols=c)
        self.declare_partials('P_dc', ['I', 'resistivity_wire', 'T_coeff_cu', 'T_windings', 'n_slots', 'n_turns', 'stack_length', 'r_strand'], rows=r, cols=c)
        self.declare_partials('P_ac', ['AC_power_factor', 'I', 'resistivity_wire', 'T_coeff_cu', 'T_windings', 'n_slots', 'n_turns', 'stack_length', 'r_strand'], rows=r, cols=c)
        self.declare_partials('P_wire', ['I', 'resistivity_wire', 'T_coeff_cu', 'T_windings', 'n_slots', 'n_turns', 'stack_length', 'r_strand', 'AC_power_factor'], rows=r, cols=c)


    def compute(self, inputs, outputs):
        rpm = inputs['rpm']
        n_m = inputs['n_m']
        mu_o = inputs['mu_o']
        mu_r = inputs['mu_r']
        r_strand = inputs['r_strand']
        T_windings = inputs['T_windings']
        T_coeff_cu = inputs['T_coeff_cu']
        resistivity_wire = inputs['resistivity_wire']
        I = inputs['I']
        stack_length = inputs['stack_length']
        n_slots = inputs['n_slots']
        n_turns = inputs['n_turns']
        n_strands = inputs['n_strands']
        AC_pf = inputs['AC_power_factor']

        outputs['f_e']              = n_m / 2 * rpm / 60                                            # Eqn 1.5 "Brushless PM Motor Design" by D. Hansleman                                       
        outputs['r_litz']           = (np.sqrt(n_strands) * 1.154 * r_strand*2)/2                   # New England Wire
        outputs['L_wire']           = (n_slots/3 * n_turns) * (stack_length*2 + .017*2)              
        outputs['temp_resistivity'] = (resistivity_wire * (1 + T_coeff_cu*(T_windings-20)))         # Eqn 4.14 "Brushless PM Motor Design" by D. Hansleman
        outputs['A_cu']             = n_turns * n_strands * 2 * np.pi * r_strand**2
        outputs['R_dc']             = outputs['temp_resistivity'] * outputs['L_wire'] / ((np.pi*(r_strand)**2)*41)
        outputs['skin_depth']       = np.sqrt( outputs['temp_resistivity'] / (np.pi * outputs['f_e'] * mu_r * mu_o) )
        outputs['P_dc']             = (I*np.sqrt(2))**2 * (outputs['R_dc']) *3/2
        outputs['P_ac']             = AC_pf * outputs['P_dc']
        outputs['P_wire']           = outputs['P_dc'] + outputs['P_ac']

    def compute_partials(self, inputs, J):
        rpm = inputs['rpm']
        n_m = inputs['n_m']
        mu_o = inputs['mu_o']
        mu_r = inputs['mu_r']
        r_strand = inputs['r_strand']
        T_windings = inputs['T_windings']
        T_coeff_cu = inputs['T_coeff_cu']
        resistivity_wire = inputs['resistivity_wire']
        I = inputs['I']
        stack_length = inputs['stack_length']
        n_slots = inputs['n_slots']
        n_turns = inputs['n_turns']
        n_strands = inputs['n_strands']
        AC_pf = inputs['AC_power_factor']

        f_e = n_m / 2 * rpm / 60  
        L_wire = (n_slots/3 * n_turns) * (stack_length*2 + .017*2) 
        temp_resistivity = (resistivity_wire * (1 + T_coeff_cu*(T_windings-20)))
        R_dc = temp_resistivity * L_wire / ((np.pi*(r_strand)**2)*41)
        P_dc = (I*np.sqrt(2))**2 * R_dc *3/2
        P_ac = AC_pf * P_dc

        d_f_e__d_n_m = J['f_e', 'n_m'] = 1 / 2 * rpm / 60
        d_f_e__d_rpm = J['f_e', 'rpm'] = n_m / 2 * 1 / 60 

        J['r_litz', 'n_strands'] = (n_strands**-.5 * 1.154 * r_strand*2)/4 
        J['r_litz', 'r_strand'] = (np.sqrt(n_strands) * 1.154 * 2)/2 

        d_L_wire__d_n_slots = J['L_wire', 'n_slots'] = (1/3 * n_turns) * (stack_length*2 + .017*2)
        d_L_wire__d_n_turns = J['L_wire', 'n_turns'] = (n_slots/3) * (stack_length*2 + .017*2)
        d_L_wire__d_stack_length = J['L_wire', 'stack_length'] = (n_slots/3 * n_turns) * 2

        d_temp_resistivity__d_resistivity_wire = J['temp_resistivity', 'resistivity_wire'] = 1 + T_coeff_cu*(T_windings-20)
        d_temp_resistivity__dT_coeff_cu = J['temp_resistivity', 'T_coeff_cu'] = resistivity_wire * (T_windings-20)
        d_temp_resistivity__dT_windings = J['temp_resistivity', 'T_windings'] = resistivity_wire * T_coeff_cu  

        d_R_dc__d_resistivity_wire = J['R_dc', 'resistivity_wire'] = L_wire/((np.pi*(r_strand)**2)*41) * d_temp_resistivity__d_resistivity_wire
        d_R_dc__d_T_coeff_cu =J['R_dc', 'T_coeff_cu'] = L_wire/((np.pi*(r_strand)**2)*41) * d_temp_resistivity__dT_coeff_cu
        d_R_dc__d_T_windings =J['R_dc', 'T_windings'] = L_wire/((np.pi*(r_strand)**2)*41) * d_temp_resistivity__dT_windings
        d_R_dc__d_n_slots =J['R_dc', 'n_slots'] = temp_resistivity / ((np.pi*(r_strand)**2)*41) * d_L_wire__d_n_slots
        d_R_dc__d_n_turns =J['R_dc', 'n_turns'] = temp_resistivity / ((np.pi*(r_strand)**2)*41) * d_L_wire__d_n_turns
        d_R_dc__d_stack_length =J['R_dc', 'stack_length'] = temp_resistivity / ((np.pi*(r_strand)**2)*41) * d_L_wire__d_stack_length
        d_R_dc__d_r_strand =J['R_dc', 'r_strand'] = -2 * temp_resistivity * L_wire / ((np.pi*(r_strand)**3)*41)

        J['skin_depth', 'n_m'] = .5*(temp_resistivity / (np.pi*f_e*mu_r*mu_o))**-0.5*temp_resistivity*-1*f_e**-2/(np.pi*mu_r*mu_o) * d_f_e__d_n_m
        J['skin_depth', 'rpm'] = .5*(temp_resistivity / (np.pi*f_e*mu_r*mu_o))**-0.5*temp_resistivity*-1*f_e**-2/(np.pi*mu_r*mu_o) * d_f_e__d_rpm
        J['skin_depth', 'resistivity_wire'] = .5*(temp_resistivity / (np.pi*f_e*mu_r*mu_o))**-0.5*1/(np.pi*f_e*mu_r*mu_o) * d_temp_resistivity__d_resistivity_wire
        J['skin_depth', 'T_coeff_cu'] = .5*(temp_resistivity / (np.pi*f_e*mu_r*mu_o))**-0.5*1/(np.pi*f_e*mu_r*mu_o) * d_temp_resistivity__dT_coeff_cu
        J['skin_depth', 'T_windings'] = .5*(temp_resistivity / (np.pi*f_e*mu_r*mu_o))**-0.5*1/(np.pi*f_e*mu_r*mu_o) * d_temp_resistivity__dT_windings
        J['skin_depth', 'mu_o'] = .5*(temp_resistivity / (np.pi*f_e*mu_r*mu_o))**-0.5*temp_resistivity*-1*mu_o**-2/(np.pi*mu_r*f_e)
        J['skin_depth', 'mu_r'] = .5*(temp_resistivity / (np.pi*f_e*mu_r*mu_o))**-0.5*temp_resistivity*-1*mu_r**-2/(np.pi*f_e*mu_o)

        J['A_cu', 'n_turns'] = n_strands * 2 * np.pi * r_strand**2
        J['A_cu', 'n_strands'] = n_turns * 2 * np.pi * r_strand**2
        J['A_cu', 'r_strand'] = n_turns * n_strands * 4 * pi * r_strand

        d_P_dc__d_I = J['P_dc', 'I'] = 2*(I*np.sqrt(2)) * (R_dc) *3/2 * np.sqrt(2)
        d_P_dc__d_resistivity_wire = J['P_dc', 'resistivity_wire'] = (I*np.sqrt(2))**2 *3/2 * d_R_dc__d_resistivity_wire
        d_P_dc__d_T_coeff_cu = J['P_dc', 'T_coeff_cu'] = (I*np.sqrt(2))**2 *3/2 * d_R_dc__d_T_coeff_cu
        d_P_dc__d_T_windings = J['P_dc', 'T_windings'] = (I*np.sqrt(2))**2 *3/2 * d_R_dc__d_T_windings
        d_P_dc__d_n_slots = J['P_dc', 'n_slots'] = (I*np.sqrt(2))**2 *3/2 * d_R_dc__d_n_slots
        d_P_dc__d_n_turns = J['P_dc', 'n_turns'] = (I*np.sqrt(2))**2 *3/2 * d_R_dc__d_n_turns
        d_P_dc__d_stack_length = J['P_dc', 'stack_length'] = (I*np.sqrt(2))**2 *3/2 * d_R_dc__d_stack_length
        d_P_dc__d_r_strand = J['P_dc', 'r_strand'] = (I*np.sqrt(2))**2 *3/2 * d_R_dc__d_r_strand

        d_P_ac__d_AC_pf = J['P_ac', 'AC_power_factor'] = P_dc
        d_P_ac__d_I = J['P_ac', 'I'] = AC_pf * d_P_dc__d_I
        d_P_ac__d_resistivity_wire =J['P_ac', 'resistivity_wire'] = AC_pf * d_P_dc__d_resistivity_wire
        d_P_ac__d_T_coeff_cu = J['P_ac', 'T_coeff_cu'] = AC_pf * d_P_dc__d_T_coeff_cu
        d_P_ac__d_T_windings = J['P_ac', 'T_windings'] = AC_pf * d_P_dc__d_T_windings
        d_P_ac__d_n_slots = J['P_ac', 'n_slots'] = AC_pf * d_P_dc__d_n_slots
        d_P_ac__d_n_turns = J['P_ac', 'n_turns'] = AC_pf * d_P_dc__d_n_turns
        d_P_ac__d_stack_length = J['P_ac', 'stack_length'] = AC_pf * d_P_dc__d_stack_length
        d_P_ac__d_r_strand = J['P_ac', 'r_strand'] = AC_pf * d_P_dc__d_r_strand

        J['P_wire', 'I'] = d_P_dc__d_I + d_P_ac__d_I
        J['P_wire', 'resistivity_wire'] = d_P_dc__d_resistivity_wire + d_P_ac__d_resistivity_wire
        J['P_wire', 'T_coeff_cu'] = d_P_dc__d_T_coeff_cu + d_P_ac__d_T_coeff_cu
        J['P_wire', 'T_windings'] = d_P_dc__d_T_windings + d_P_ac__d_T_windings
        J['P_wire', 'n_slots'] = d_P_dc__d_n_slots + d_P_ac__d_n_slots
        J['P_wire', 'n_turns'] = d_P_dc__d_n_turns + d_P_ac__d_n_turns
        J['P_wire', 'stack_length'] = d_P_dc__d_stack_length + d_P_ac__d_stack_length
        J['P_wire', 'r_strand'] = d_P_dc__d_r_strand + d_P_ac__d_r_strand
        J['P_wire', 'AC_power_factor'] = d_P_ac__d_AC_pf

class SteinmetzLossComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('f_e', 900*np.ones(nn), units='Hz', desc='Electrical frequency')
        self.add_input('B_pk', 2.05, units='T', desc='Peak magnetic field in Tesla')
        self.add_input('alpha_stein', 1.286, desc='Alpha coefficient for steinmetz, constant')
        self.add_input('beta_stein', 1.76835, desc='Beta coefficient for steinmentz, dependent on freq')  
        self.add_input('k_stein', 0.0044, desc='k constant for steinmentz')
        self.add_input('sta_mass', 1, units='kg', desc='total mass of back-iron')

        self.add_output('P_steinmetz', 200*np.ones(nn), units='W', desc='Simplified steinmetz losses')

        r = c = np.arange(nn)  # for scalar variables only
        self.declare_partials('P_steinmetz', ['k_stein', 'f_e', 'alpha_stein', 'B_pk', 'beta_stein', 'sta_mass'], rows=r, cols=c)

    def compute(self, inputs, outputs):
        f_e = inputs['f_e']
        B_pk = inputs['B_pk']
        alpha_stein = inputs['alpha_stein']
        beta_stein = inputs['beta_stein']
        k_stein = inputs['k_stein']
        sta_mass = inputs['sta_mass']

        outputs['P_steinmetz'] = k_stein * f_e**alpha_stein * B_pk**beta_stein * sta_mass

    def compute_partials(self, inputs, J):
        f_e = inputs['f_e']
        B_pk = inputs['B_pk']
        alpha_stein = inputs['alpha_stein']
        beta_stein = inputs['beta_stein']
        k_stein = inputs['k_stein']
        sta_mass = inputs['sta_mass']

        J['P_steinmetz', 'k_stein'] = f_e**alpha_stein * B_pk**beta_stein * sta_mass
        J['P_steinmetz', 'f_e'] = alpha_stein*k_stein * f_e**(alpha_stein-1) * B_pk**beta_stein * sta_mass
        J['P_steinmetz', 'alpha_stein'] = k_stein * f_e**alpha_stein * B_pk**beta_stein * sta_mass * np.log(f_e)
        J['P_steinmetz', 'B_pk'] = k_stein * f_e**alpha_stein * B_pk**(beta_stein-1) * sta_mass*beta_stein
        J['P_steinmetz', 'beta_stein'] = k_stein * f_e**alpha_stein * B_pk**beta_stein * sta_mass * np.log(B_pk)
        J['P_steinmetz', 'sta_mass'] = k_stein * f_e**alpha_stein * B_pk**beta_stein
        



