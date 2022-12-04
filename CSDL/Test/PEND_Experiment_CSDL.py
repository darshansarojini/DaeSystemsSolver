# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 11:39:27 2022

@author: Zeyu Huang
"""

import sys
sys.path.append('..')
from Model.PEND_Model_CSDL import PEND_R,PEND_F
from ForSol.Time_Marching_CSDL import Newton
from AdSol.Adjoint_Solver_CSDL import Adjoint
from csdl_om import Simulator

import numpy as np
import matplotlib.pyplot as plt

model_R = PEND_R()
model_F = PEND_F()
x0 = np.array([[0,3]])
xDot0 = np.array([[3,0]])
t_initial = 0
t_final = 10
time_step = 10**(-3)
tolerance = 10**(-8)
mu = ['L','b','m']

import timeit
start = timeit.default_timer()

sim_R = Simulator(model_R)
experiment = Newton(sim_R,x0,xDot0,t_initial,t_final,time_step,tolerance)

stop = timeit.default_timer()
print('Time: ', stop - start) 

plt.plot(experiment.time,experiment.x_hist[:,0],'k-',linewidth=2)
plt.plot(experiment.time,experiment.x_hist[:,1],'b--',linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Displacement (rad) and Velocity(rad/s)')
plt.legend(['Angular Displacement','Angular Velocity'])
plt.grid(True)
plt.show()
