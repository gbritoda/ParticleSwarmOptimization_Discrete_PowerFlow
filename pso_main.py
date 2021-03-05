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

# def run_pso(net, pso_params):
### DEFINING VARIABLES. TBD ADD THIS TO FUNCTION
shunt_values = net_params['shunt_values'][str(nb)]
v_max = net.bus.max_vm_pu.to_numpy(dtype = np.float32)[:ng]
v_min = net.bus.min_vm_pu.to_numpy(dtype = np.float32)[:ng]
tap_max = np.repeat(net_params['tap_values'][-1], nt)
tap_min = np.repeat(net_params['tap_values'][0], nt)
shunt_max = shunt_values[:,-1]
shunt_min = shunt_values[:, 0]

# independent variables
# Voltages
v_temp = net.gen.vm_pu.to_numpy(dtype=np.float32)
#  tap_pu = (tap_pos + tap_neutral)*tap_step_percent/100 (PandaPower)
taps_temp = 1 + ((net.trafo.tap_pos.to_numpy(dtype = np.float32)[0:nt] +\
                      net.trafo.tap_neutral.to_numpy(dtype = np.float32)[0:nt]) *\
                     (net.trafo.tap_step_percent.to_numpy(dtype = np.float32)[0:nt]/100))
#  shunt_pu = -100*shunt (PandaPower)
shunt_temp = -net.shunt.q_mvar.to_numpy(dtype = np.float32)/100

# Particles will be in the following format:
x = np.array([np.concatenate((v_temp, taps_temp, shunt_temp),axis=0)])
x = np.squeeze(x)

nVar = len(x)
upper_bounds = np.squeeze(np.array([np.concatenate((v_max, tap_max, shunt_max),axis=0)]))
lower_bounds = np.squeeze(np.array([np.concatenate((v_min, tap_min, shunt_min),axis=0)]))

def fitness_function():
    # Placeholder
    return True

problem = {
        'CostFunction': fitness_function,
        'nVar': 8,
        'VarMin': np.array(lb),
        'VarMax': np.array(ub)
}