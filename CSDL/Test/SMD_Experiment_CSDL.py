# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 11:39:27 2022

@author: Zeyu Huang
"""

import sys
sys.path.append('..')
from Model.SMD_Model_CSDL import SMD_R
from ForSol.Time_Marching import Newton

import numpy as np
import matplotlib.pyplot as plt

sim_R = SMD_R()
x0 = np.array([[1],[0]])
xDot0 = np.array([[0],[0]])
t_initial = 0
t_final = 10
time_step = 10**(-3)
tolerance = 10**(-8)

experiment = Newton(sim_R,x0,xDot0,t_initial,t_final,time_step,tolerance)

plt.plot(experiment.time,experiment.x_hist[0,:])
plt.title('Mass Movement')
plt.xlabel('time (s)')
plt.ylabel('position (m)')
plt.grid(True)
plt.show()