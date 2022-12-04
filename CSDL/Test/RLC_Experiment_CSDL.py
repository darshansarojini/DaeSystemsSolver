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

model_R = RLC_R()
model_F = RLC_F()
x0 = np.array([[0,0,0,2.4]])
xDot0 = np.array([[0,0,0,0]])
t_initial = 0
t_final = 12
time_step = 10**(-3)
tolerance = 10**(-8)
mu = ['RE','L','C','V']

sim_R = Simulator(model_R)
experiment = Newton(sim_R,x0,xDot0,t_initial,t_final,time_step,tolerance)

plt.plot(experiment.time,experiment.x_hist[:,0],'k-',linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.grid(True)
plt.show()
