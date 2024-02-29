# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 11:39:27 2022

@author: Zeyu Huang
"""

import sys
sys.path.append('..')
from dae_systems_solver.CSDL.Model.SMD_Model_CSDL import SMD_R,SMD_F
from dae_systems_solver.CSDL.ForSol.Time_Marching_CSDL import Newton
from dae_systems_solver.CSDL.AdSol.Adjoint_Solver_CSDL import Adjoint
from python_csdl_backend import Simulator

import numpy as np
import matplotlib.pyplot as plt

# experiment initialization
model_R = SMD_R()
model_F = SMD_F()
x0 = np.array([[1,0]])
xDot0 = np.array([[0,0]])
t_initial = 0
t_final = 10
time_step = 10**(-3)
tolerance = 10**(-8)
mu = ['c','m','k']

# solve state variables implementing forward solver
sim_R = Simulator(model_R) 
experiment = Newton(sim_R,x0,xDot0,t_initial,t_final,time_step,tolerance)

# plot the results
plt.plot(experiment.time,experiment.x_hist[:,0],'k-',linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.grid(True)
plt.show()

"""
compute gradients using adjoint method
"""
smd_adj = Adjoint(model_R,model_F,experiment.time,experiment.x_hist,experiment.xDot_hist,mu)
grad = smd_adj.final_grad()

"""
compute gradients using finite element method
"""
k = 5.0
# initial condition
val = 0
for i in range(len(experiment.time)):
    xsq = experiment.x_hist[i,0]*experiment.x_hist[i,0]
    val = val + 0.5*k*experiment.time_step*xsq

# modify and update design variables
k = 5.0 + 10**(-4)
sim_R = Simulator(model_R)
sim_R['k'] = k
sim_R.run()

# update solution of modified DAE system
experiment = Newton(sim_R,x0,xDot0,t_initial,t_final,time_step,tolerance)
val_plus = 0
for i in range(len(experiment.time)):
    xsq = experiment.x_hist[i,0]*experiment.x_hist[i,0]
    val_plus = val_plus + 0.5*k*experiment.time_step*xsq

"""
compare results
"""
fd_adjoint = (val_plus-val)/10**(-4)
dfdk = grad[2]
print(fd_adjoint)
