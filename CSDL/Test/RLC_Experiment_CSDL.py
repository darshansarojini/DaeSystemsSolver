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

import numpy as np
import matplotlib.pyplot as plt

sim_R = RLC_R()
sim_F = RLC_F()
x0 = np.array([[0,0,0,2.4]])
xDot0 = np.array([[0,0,0,0]])
t_initial = 0
t_final = 12
time_step = 10**(-3)
tolerance = 10**(-8)
mu = ['R','L','C','V']

import timeit
start = timeit.default_timer()

experiment = Newton(sim_R,x0,xDot0,t_initial,t_final,time_step,tolerance)

stop = timeit.default_timer()
print('Time: ', stop - start) 

plt.plot(experiment.time,experiment.x_hist[:,0])
plt.title('Current vs time')
plt.xlabel('time (s)')
plt.ylabel('Current (A)')
plt.grid(True)
plt.show()
