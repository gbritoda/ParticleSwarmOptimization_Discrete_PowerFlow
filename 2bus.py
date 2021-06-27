from yarpiz.pso import PSO
import numpy as np
# import pandapower as pp
# import pandapower.networks
import random
import pandas as pd
import matplotlib.pyplot as plt
from math import cos,sin,pi
import time

random.seed(time.time())

def con1(x):
    v1, v2, th = x
    return 0.4 + 0.1923*(v2**2) -v1*v2*(0.1923*cos(th)-0.9615*sin(th))


def con2(x):
    v1, v2, th = x
    # <= 0
    return min(0,0.15 -(-(v2**2)*0.9 - v2*v1*(0.1923*sin(th)-0.9*cos(th))))

def penalty(x):

    assert len(ub) == len(lb)
    assert len(ub) != 0

    eq = 0
    ineq = 0

    # Equality and inequalities penalties
    for eq_con in eq_f:
        eq += eq_con(x)**2
    for ineq_con in ineq_f:
        ineq += ineq_con(x)**2

    # Boundaries penalties
    ## As of now there are no boundary penalties,
    # since the algorithm takes care that the particles are inside the boundaries
    # temp = np.array(x) + np.array(lb)
    # lpen = np.dot(temp,np.less(x,lb))
    # upen = np.dot(x,np.greater(x,ub))
    # print("lpen {} upen {}".format(lpen,upen))
    # return (eq+ineq+np.abs(upen) + np.abs(lpen))*1000
    return (eq+ineq)*10000

def fitness(x):
    """ Function to be minimized """
    v1, v2, th = x
    return 0.1923*(v1**2 + v2**2 - 2*v1*v2*cos(th)) + penalty(x)

def print_summary(fopt, xopt, idx, pnlty, elapsed):
    run_no = idx+1
    msg =   "\nBest run was run %(run_no)s \n" +\
            " Cost = %(fopt).12f \n" +\
            " Penalty = %(pnlty).12f \n" +\
            " Population = %(xopt)s \n" +\
            " Elapsed = %(elapsed).2f s \n"
    print(msg % locals())

#### MAIN PROGRAM ####
global eq_f, ineq_f, ub, lb

eq_f = [con1]
ineq_f = [con2]

# Upper and lower bounds
ub = [1.1,1.1, pi]
lb = [0.95,0.95,-pi]

problem = {
            'CostFunction': fitness,
            'nVar': 3,
            'VarMin': np.array(lb),
            'VarMax': np.array(ub),
}

parameters = {
    'MaxIter':  200,
    'PopSize':  75,
    'c1':       1.5,
    'c2':       2,
    'w':        1,
    'wdamp':    0.995
}

test_params = {
    'Runs': 100,
    'Plot': True
}

penalties = []
fitness = []
x_runs = []
times = []
particles = []

for run in range(test_params['Runs']):
    print("Run number {}".format(run))
    start = time.time()

    # gbest, pop = PSO(problem, MaxIter = 200, PopSize = 70, c1 = 1.5, c2 = 2, w = 1, wdamp = 0.995)
    gbest, pop, plot_pop = PSO(problem, **parameters)

    times.append(time.time() - start)
    penalties.append(penalty(gbest['position']))
    x_runs.append(gbest['position'])
    fitness.append(gbest['cost'] - penalty(gbest['position']))

fopt = min(fitness)
idx = fitness.index(fopt)
xopt = x_runs[idx]
pnlty = penalties[idx]
elapsed = times[idx]

print_summary(fopt, xopt, idx, pnlty, elapsed)
exit(0)
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

th = -0.36586622
# th = xopt[2]
v1 = np.arange(0.95,1.1,0.01)
v2 = np.arange(0.95,1.1,0.01)
v1,v2 = np.meshgrid(v1,v2)

Z=0.1923*(v1**2 + v2**2 - 2*v1*v2*cos(th))

# Plot the surface
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(v1,v2,Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(0.03064, 0.04)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter('{x:.02f}')
fig.colorbar(surf, shrink=0.5, aspect=5)

# scatter plot
_p = np.array([p['position'] for p in pop])
v1s = _p[:,0]
v2s = _p[:,1]
zs=0.1923*(v1s**2 + v2s**2 - 2*v1s*v2s*cos(th))
ax.scatter(v1s, v2s, zs, marker='o')

plt.show()