from __future__ import absolute_import
import numpy as np
from math import pi
from openmdao.api import Problem, IndepVarComp, ExplicitComponent, ExecComp
from openmdao.api import NewtonSolver, Group, DirectSolver, NonlinearRunOnce, LinearRunOnce, view_model, BalanceComp
from volume_group import motor_size, motor_mass
from thermal_group import motor_losses
from em_group import Efficiency, torque



if __name__ == "__main__":
    p = Problem()
    model = p.model

    ind = model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])

    ind.add_output('v', val=385, units='V', desc='RMS voltage')
    ind.add_output('rpm', val=5400, units='rpm', desc='Rotation speed')

    ind.add_output('k', val=0.94)                           # Stacking factor
    ind.add_output('k_wb', val=0.50)                        # copper fill factor
    ind.add_output('gap', val=0.001, units='m')             # Air gap distance
    ind.add_output('n', val=12)                             # Number of wire turns     
    ind.add_output('i', val=34.8, units='A')                # RMS Current
    ind.add_output('radius_motor', val=0.0795, units='m')   # Motor outer radius
    ind.add_output('t_mag', val=.0044, units='m')           # Magnet thickness

    ind.add_output('b_g', val = 1, units='T')           # Air gap flux Density    !! Flux values may represent 100% slot fill !!
    ind.add_output('b_ry', val=3, units='T')            # Rotor yoke flux density
    ind.add_output('b_sy', val=3, units='T')            # Stator yoke flux density
    ind.add_output('b_t', val=3, units='T')             # Tooth Flux Density

    ind.add_output('n_s', val=21)                # Number of Slots
    ind.add_output('n_m', val=20)                # Number of magnets

    ind.add_output('stack_length', val=0.0345, units='m')   # Stack Length
    ind.add_output('rho', val=7845, units='kg/m**3')        # Density of Hiperco-50
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
    
    model.add_subsystem('size', motor_size(), promotes_inputs=['n','i','k_wb','radius_motor','gap','rot_or','b_g','k','b_ry','n_m','b_sy','b_t','n_s', 't_mag'], promotes_outputs=['w_ry', 'w_sy', 'w_t','s_d','rot_ir','sta_ir'])
    model.add_subsystem('losses', motor_losses(), promotes_inputs=['f_e', 'B_pk', 'K_h', 'K_e', 'K_h_alpha', 'K_h_beta'], promotes_outputs=['P_e','P_h'])
    model.add_subsystem(name='balance', subsys=bal)
    model.add_subsystem('mass', motor_mass(), promotes_inputs=['t_mag','rho_mag','rho','radius_motor','n_s','sta_ir','w_t','stack_length','s_d','rot_or','rot_ir'], promotes_outputs=['sta_mass','rot_mass','mag_mass'])
    model.add_subsystem('torque', torque(), promotes_inputs=['rot_or','b_g','i','n_m','n','stack_length'], promotes_outputs=['tq'])
    model.add_subsystem('mot_eff', Efficiency(), promotes_inputs=['rpm', 'i','tq','v'], promotes_outputs=['P_in', 'P_out'])

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
    print('Power In  ........................',  p.get_val('P_in'))
    print('Power out  .......................',  p.get_val('P_out'))    
