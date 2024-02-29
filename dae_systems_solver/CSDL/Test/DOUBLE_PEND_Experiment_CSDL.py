# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 11:39:27 2022

@author: Zeyu Huang
"""

import sys
sys.path.append('..')
from dae_systems_solver.CSDL.Model.DOUBLE_PEND_Model_CSDL import DOUBLE_PEND_R,DOUBLE_PEND_F
from dae_systems_solver.CSDL.ForSol.Time_Marching_CSDL import Newton
from dae_systems_solver.CSDL.AdSol.Adjoint_Solver_CSDL import Adjoint
# from csdl_om import Simulator
from python_csdl_backend import Simulator

import numpy as np
import matplotlib.pyplot as plt

# experiment initialization
model_R = DOUBLE_PEND_R()
model_F = DOUBLE_PEND_F()
x0 = np.array([1., 0., np.pi/2, 1., -1., 0.,
                0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]).reshape((1,20))
xDot0 = np.zeros((20,)).reshape((1,20))
t_initial = 0
# t_final = 2
t_final = 5
# time_step = 10**(-3)
time_step = 0.01
# time_step = 0.5
tolerance = 10**(-8)
mu = ['l1', 'l2','b','m1', 'm2']

# solve state variables implementing forward solver
sim_R = Simulator(model_R)
experiment = Newton(sim_R,x0,xDot0,t_initial,t_final,time_step,tolerance)

# plot the results
plt.figure()
plt.plot(experiment.time,experiment.x_hist[:,2],'k-',linewidth=2, label='Angular Displacement 1')
plt.plot(experiment.time,experiment.x_hist[:,5],'b--',linewidth=2, label='Angular Displacement 2')
plt.xlabel('Time (s)')
plt.ylabel('$\\theta_{1}$,$\\theta_{2}$')
plt.legend()
plt.title('Angular Displacements')
plt.grid(True)

plt.figure()
plt.plot(experiment.time,experiment.x_hist[:,12],'k-',linewidth=2, label='Angular Velocity 1')
plt.plot(experiment.time,experiment.x_hist[:,15],'b--',linewidth=2, label='Angular Velocity 2')
plt.xlabel('Time (s)')
plt.ylabel('$\dot{\\theta_{1}}$,$\dot{\\theta_{2}}$')
plt.legend()
plt.title('Angular Velocities')
plt.grid(True)

plt.figure()
plt.plot(experiment.time,experiment.xDot_hist[:,12],'k-',linewidth=2, label='Angular Acceleration 1')
plt.plot(experiment.time,experiment.xDot_hist[:,15],'b--',linewidth=2, label='Angular Acceleration 2')
plt.xlabel('Time (s)')
plt.ylabel('$\ddot{\\theta_{1}}$,$\ddot{\\theta_{2}}$')
plt.legend()
plt.title('Angular Accelerations')
plt.grid(True)

plt.figure()
plt.plot(experiment.x_hist[:,0],experiment.x_hist[:,1],'k-',linewidth=2, label='Mass 1')
plt.plot(experiment.x_hist[:,3],experiment.x_hist[:,4],'b--',linewidth=2, label='Mass 2')
plt.xlabel('$x_{1},x_{2}$')
plt.ylabel('$y_{1},y_{2}$')
plt.legend()
plt.grid(True)
plt.title('Mass Trajectories')

plt.figure()
plt.plot(experiment.time,experiment.x_hist[:,6],'k-',linewidth=2, label='Lagrange Multiplier 1')
plt.plot(experiment.time,experiment.x_hist[:,7],'b--',linewidth=2, label='Lagrange Multiplier 2')
plt.xlabel('Time (s)')
plt.ylabel('$\lambda_{1}$,$\lambda_{2}$')
plt.legend()
plt.title('Lagrange Multipliers For Mass 1')
plt.grid(True)

plt.figure()
plt.plot(experiment.time,experiment.x_hist[:,8],'k-',linewidth=2, label='Lagrange Multiplier 3')
plt.plot(experiment.time,experiment.x_hist[:,9],'b--',linewidth=2, label='Lagrange Multiplier 4')
plt.xlabel('Time (s)')
plt.ylabel('$\lambda_{3}$,$\lambda_{4}$')
plt.legend()
plt.title('Lagrange Multipliers For Mass 2')
plt.grid(True)

plt.show()

"""
compute gradients using adjoint method
"""
pend_adj = Adjoint(model_R,model_F,experiment.time,experiment.x_hist,experiment.xDot_hist,mu)
grad = pend_adj.final_grad()

"""
compute gradients using finite difference method
"""
# initial condition
l1 = 1.
l2 = 1.
m1 = 1.
m2 = 1.
val = 0
for i in range(len(experiment.time)):
    val = val + 0.5*m1*((experiment.x_hist[i,10])**2 + (experiment.x_hist[i,11])**2)*l1**2 + \
        0.5*m2*((experiment.x_hist[i,16])**2 + (experiment.x_hist[i,17])**2)*l2**2

# modify and update design variables
l1 = 1.0 + 10**(-4)
sim_R = Simulator(model_R)
sim_R['l1'] = l1
sim_R.run()

# update solution of modified DAE system
experiment = Newton(sim_R,x0,xDot0,t_initial,t_final,time_step,tolerance)
val_plus = 0
for i in range(len(experiment.time)):
    val_plus = val_plus + 0.5*m1*((experiment.x_hist[i,10])**2 + (experiment.x_hist[i,11])**2)*l1**2 + \
        0.5*m2*((experiment.x_hist[i,16])**2 + (experiment.x_hist[i,17])**2)*l2**2

"""
compare results
"""
fd_adjoint = (val_plus-val)/10**(-4)
dfdk = grad[0]
print(fd_adjoint)
print(dfdk)
print(100*(dfdk-fd_adjoint)/dfdk)

