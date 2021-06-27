# from yarpiz.pso import PSO
# import yarpiz.pso as yp
import argparse
import numpy as np
import pandapower as pp
from pprint import pprint
import matplotlib.pyplot as plt
from pandapower.networks import case14, case_ieee30, case118
import time
from lib import yarpiz_custom_pso as yp
import lib.fpor_tools as fpor

parser = argparse.ArgumentParser(description='Particle Swarm Optimization')
parser.add_argument('-nb', '--num_bus', type=int, help="Bus number", default=14)
parser.add_argument('-r', '--runs', type=int, help="Number of runs", default=1)
parser.add_argument('-p', '--plot', action='store_true', help="Plot the results")
args = parser.parse_args()

global net, net_params
net = {
    14:case14,
    30:case_ieee30,
    118:case118
# Change the number below to select the case.
}[args.num_bus]()
print('\nIEEE System {} bus\n'.format(args.num_bus))
print(net)
net_params = fpor.network_set(net)

global nb, ng, nt, ns
nb = net_params['n_bus']
ng = net_params['n_gen']
nt = net_params['n_taps']
ns = net_params['n_shunt']
pp.runpp(net, algorithm = 'nr', numba = True)

pso_params = {
    'MaxIter':  75,
    'PopSize':  70,
    'c1':       1.5,
    'c2':       2,
    'w':        1,
    'wdamp':    0.995
}

test_params = {
    'Runs': args.runs,
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
    # TBD Description

    x = fpor.run_power_flow(x, net, net_params, ng, nt, ns)
    # fopt and boundaries penalty
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
conv_plot = []
results = []
for run in range(1,test_params['Runs']+1):
    print('Run No {} out of {}'.format(run,test_params['Runs']))
    start = time.time()
    gbest, pop, convergence_points = yp.PSO(problem, **pso_params)
    elapsed = round(time.time() - start, 2)
    print('Run No {} results:'.format(run))
    results.append(\
        fpor.debug_fitness_function(gbest['position'],net,net_params,test_params,elapsed))
    if args.plot:
        conv_plot.append(convergence_points)

fopt_values = [results[i]['f'] for i in range(len(results))]
ind = np.argmin(fopt_values)
best_result = results[ind]
print("\nFinal results of best run: (Run {})".format(ind+1))
# pprint(best_result, sort_dicts=False)
pprint(best_result)

print("\nStatistics:")
pprint(fpor.get_results_statistics(fopt_values))

print("\nTest Parameters:")
pprint(test_params)

print("\nPSO Parameters:")
pprint(pso_params)

if args.plot:
    fpor.plot_results(nb, best_result, voltage_plot=True)
    fpor.plot_results(nb, best_result, voltage_plot=False)
    fpor.plot_convergence(nb, conv_plot[ind])
