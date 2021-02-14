from yarpiz.pso import PSO
import numpy as np
import pandapower as pp
from pandapower.networks import case14, case_ieee30

import lib.fpor_tools as fpor

net = case14()
# pp.runpp(rede, algorithm = 'nr', numba = True)

net_params = fpor.network_set(net)
nb = net_params['n_bus']
ng = net_params['n_gen']
nt = net_params['n_taps']
ns = net_params['n_shunt']

pso_params = {
    'MaxIter':  200,
    'PopSize':  70,
    'c1':       1.5,
    'c2':       2,
    'w':        1,
    'wdamp':    0.995
}

test_params = {
    'Runs': 5
}

def run_pso(net, pso_params):

    shunt_values = net_params['shunt_values'][str(nb)]
    v_max = net.bus.max_vm_pu.to_numpy(dtype = np.float32)[:ng]
    v_min = net.bus.min_vm_pu.to_numpy(dtype = np.float32)[:ng]
    tap_max = np.repeat(net_params['tap_values'][-1], nt)
    tap_min = np.repeat(net_params['tap_values'][0], nt)
    
    shunt_max = shunt_values[:,-1]
    shunt_min = shunt_values[:, 0]


def main():
    return True