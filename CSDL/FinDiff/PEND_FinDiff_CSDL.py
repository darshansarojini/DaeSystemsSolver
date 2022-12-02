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
mu = ['l','b','m']
l = 1.
m = 1.

sim_R = Simulator(model_R)
experiment = Newton(sim_R,x0,xDot0,t_initial,t_final,time_step,tolerance)

val = 0
for i in range(len(experiment.time)):
    val = val + 0.5*m*(experiment.x_hist[i,1]*l)**2

pend_adj = Adjoint(model_R,model_F,experiment.time,experiment.x_hist,experiment.xDot_hist,mu)

l = 1.0 + 10**(-4)
sim_R = Simulator(model_R)
sim_R['l'] = l
sim_R.run()

experiment = Newton(sim_R,x0,xDot0,t_initial,t_final,time_step,tolerance)

val_plus = 0
for i in range(len(experiment.time)):
    val_plus = val_plus + 0.5*m*(experiment.x_hist[i,1]*l)**2

fd_adjoint = (val_plus-val)/10**(-4)
grad = smd_adj.final_grad()
dfdk = grad[0]
print(fd_adjoint)
#fd_adjoint = 1.160820
print(dfdk)
print(100*(dfdk-fd_adjoint)/dfdk)
