# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 11:39:27 2022

@author: Zeyu Huang
"""

import sys
sys.path.append('..')
from Model.SMD_Model_CSDL import SMD_R,SMD_F
from ForSol.Time_Marching_CSDL import Newton
from AdSol.Adjoint_Solver_CSDL import Adjoint
from csdl_om import Simulator

import numpy as np
import matplotlib.pyplot as plt

model_R = SMD_R()
model_F = SMD_F()
x0 = np.array([[1,0]])
xDot0 = np.array([[0,0]])
t_initial = 0
t_final = 10
time_step = 10**(-3)
tolerance = 10**(-8)
mu = ['c','m','k']

import timeit
start = timeit.default_timer()
sim_R = Simulator(model_R) 
experiment = Newton(sim_R,x0,xDot0,t_initial,t_final,time_step,tolerance)

stop = timeit.default_timer()
print('Time: ', stop - start) 

plt.plot(experiment.time,experiment.x_hist[:,0],'k-',linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.grid(True)
plt.show()

#smd_adj = Adjoint(model_R,model_F,experiment.time,experiment.x_hist,experiment.xDot_hist,mu)

#grad = smd_adj.final_grad()
#dfdk = grad[2]
#fd_adjoint = 1.160820
#print(dfdk)
#print(100*(dfdk-fd_adjoint)/dfdk)
