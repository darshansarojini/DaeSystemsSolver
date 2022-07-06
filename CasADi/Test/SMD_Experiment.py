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
import Model.SMD_Model

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
smd_model = Model.SMD_Model.SpringMassDamper(c,m,k,x0,xDot0)
experiment = Newton(smd_model,t_initial,t_final,time_step,tolerance) 

# plot the result
plt.plot(experiment.time,experiment.x_hist[0,:])
plt.title('Mass Movement')
plt.xlabel('time (s)')
plt.ylabel('position (m)')
plt.grid(True)
plt.show()