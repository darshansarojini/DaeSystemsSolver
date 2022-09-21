# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 10:44:39 2022

@author: Vera_
"""

from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from ForSol.Time_Marching import Newton
from Model.Chem_Model import ChemReaction
# from AdSol.Adjoint_Solver import Adjoint

# initialization constants
A = 0.04
B = 3*10**7
C = 10**4
x0 = np.array([[1],[0],[0]])
xDot0 = np.array([[-0.04],[0.04],[0]])
t_initial = 0
t_final = 10
time_step = 10**(-3)
tolerance = 10**(-8)

# experiment
chem_model = ChemReaction(A,B,C,x0,xDot0,time_step)
experiment = Newton(chem_model,t_initial,t_final,time_step,tolerance) 

# plot the result
plt.plot(experiment.time,experiment.x_hist[1,:])
plt.title('Mass Movement')
plt.xlabel('time (s)')
plt.ylabel('position (m)')
plt.grid(True)
plt.show()