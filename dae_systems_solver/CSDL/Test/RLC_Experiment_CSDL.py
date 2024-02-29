# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 11:39:27 2022

@author: Zeyu Huang
"""

import sys
sys.path.append('..')
from Model.RLC_Model_CSDL import RLC_R,RLC_F
from ForSol.Time_Marching_CSDL import Newton
from AdSol.Adjoint_Solver_CSDL import Adjoint
from csdl_om import Simulator

import numpy as np
import matplotlib.pyplot as plt

# experiment initialization
model_R = RLC_R()
model_F = RLC_F()
x0 = np.array([[0,0,0,2.4]])
xDot0 = np.array([[0,0,0,0]])
t_initial = 0
t_final = 12
time_step = 10**(-3)
tolerance = 10**(-8)
mu = ['RE','L','C','V']

# solve state variables implementing forward solver
sim_R = Simulator(model_R)
experiment = Newton(sim_R,x0,xDot0,t_initial,t_final,time_step,tolerance)

# plot the results
plt.plot(experiment.time,experiment.x_hist[:,0],'k-',linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.grid(True)
plt.show()

"""
compute gradients using finite element method
"""
rlc_adj = Adjoint(model_R,model_F,experiment.time,experiment.x_hist,experiment.xDot_hist,mu)
grad = rlc_adj.final_grad()

"""
compute gradients using finite element method
"""
# initial condition
RE = 1.1
L = 1.6
C = 0.8
V = 2.4
val = 0
for i in range(len(experiment.time)):
    val = val + time_step*V*experiment.x_hist[i,0]

# modify and update design variables
RE = 1.1 + 10**(-4)
sim_R = Simulator(model_R)
sim_R['RE'] = RE
sim_R.run()

# update solution of modified DAE system
experiment = Newton(sim_R,x0,xDot0,t_initial,t_final,time_step,tolerance)
val_plus = 0
for i in range(len(experiment.time)):
    val_plus = val_plus + time_step*V*experiment.x_hist[i,0]

"""
compare results
"""
fd_adjoint = (val_plus-val)/10**(-4)
dfdRE = grad[0]
print(fd_adjoint)
print(dfdRE)
print(100*(dfdRE-fd_adjoint)/dfdRE)
