# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 13:55:10 2022

@author: Zeyu Huang
"""

from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from ForSol.Time_Marching import Newton
from Model.SMD_Model import SpringMassDamper
from AdSol.Adjoint_Solver import Adjoint

# initialization constants
m = 2.5
c = 0.2
k = 5.0
x0 = np.array([[1],[0]])
xDot0 = np.array([[0],[0]])
t_initial = 0
t_final = 10
time_step = 10**(-3)
tolerance = 10**(-8)

# experiment
smd_model = SpringMassDamper(c,m,k,x0,xDot0,time_step)

import timeit
start = timeit.default_timer()

experiment = Newton(smd_model,t_initial,t_final,time_step,tolerance) 

stop = timeit.default_timer()
print('Time: ', stop - start) 

# # plot the result
plt.plot(experiment.time,experiment.x_hist[0,:])
plt.title('Mass Movement')
plt.xlabel('time (s)')
plt.ylabel('position (m)')
plt.grid(True)
plt.show()

smd_adj = Adjoint(smd_model,experiment.time,experiment.x_hist,experiment.xDot_hist)
plt.plot(experiment.time,smd_adj.adj_hist[0,:])
plt.show()

plt.plot(experiment.time,smd_adj.adj_hist[1,:])
plt.show()

dfdk = smd_adj.final_dfdk(smd_model)
fd_adjoint = 1.160820
print(dfdk)
print(100*(dfdk-fd_adjoint)/dfdk)