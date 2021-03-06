# from yarpiz.pso import PSO
# import yarpiz.pso as yp
import numpy as np
import pandapower as pp
from pandapower.networks import case14, case_ieee30

from lib import yarpiz_custom_pso as yp
import lib.fpor_tools as fpor

global net, net_params
net = case14()
net_params = fpor.network_set(net)

global nb, ng, nt, ns
nb = net_params['n_bus']
ng = net_params['n_gen']
nt = net_params['n_taps']
ns = net_params['n_shunt']
pp.runpp(net, algorithm = 'nr', numba = True)

pso_params = {
    'MaxIter':  100,
    'PopSize':  70,
    'c1':       1.5,
    'c2':       2,
    'w':        1,
    'wdamp':    0.995
}

test_params = {
    'Runs': 1,
    'lambda_volt': 1e3,
    'lambda_tap': 1e3,
    'lambda_shunt': 1e7,
    'volt_threshold':1e-12,
    'tap_threshold': 1e-12,
    'shunt_threshold':1e-12
}

# lambda -> Multiplies discrete penalties
global lambd_volt, lambd_tap, lambd_shunt, tap_thr, sh_thr
lambd_volt = test_params['lambda_volt']
lambd_tap = test_params['lambda_tap']
lambd_shunt = test_params['lambda_shunt']
volt_thr = test_params['volt_threshold']
tap_thr = test_params['tap_threshold']
sh_thr = test_params['shunt_threshold']

def fitness_function(x):
    #TBD Description

    x = fpor.run_power_flow(x, net, net_params, ng, nt, ns)
    #fopt and boundaries penalty
    f, pen_v = fpor.fopt_and_penalty(net, net_params,n_threshold=volt_thr)
    tap_pen = fpor.senoidal_penalty_taps(x, net_params, n_threshold=tap_thr)
    shunt_pen = fpor.polinomial_penalty_shunts(x, net_params, n_threshold=sh_thr)

    return f + lambd_volt*pen_v + lambd_tap*tap_pen + lambd_shunt*shunt_pen

upper_bounds, lower_bounds = fpor.get_upper_and_lower_bounds_from_net(net, net_params)
n_var = fpor.get_variables_number_from_net(net, net_params)
problem = {
        'CostFunction': fitness_function,
        'nVar': n_var,
        'VarMin': lower_bounds,
        'VarMax': upper_bounds
}

for run in range(1,test_params['Runs']+1):
    print('Run No {}'.format(run))
    gbest, pop = yp.PSO(problem, **pso_params)
    fpor.debug_fitness_function(gbest['position'],net,net_params,test_params)
