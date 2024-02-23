# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 11:51:31 2022

@author: Zeyu Huang

This script contains an object which solves DAE systems using time-marching
method. At each time step, the solver implements Newton's method to solve
state variables.
"""

import numpy as np
from csdl_om import Simulator

class Newton:
    """
    Class constructor which computes the state variables and first order 
    differential of state variables.

    input:
        sim_R       - Simulator of CSDL residual model (for details check Model
                      and Test examples)
        x_0         - initial state
        x_Dot0      - first order differential of initial state
        t_initial   - initial time
        t_final     - final time
        time_step   - length of each time marching steps
        tolerance   - maximum error of Newton's method
    """
    def __init__(self,sim_R,x0,xDot0,t_initial,t_final,time_step,tolerance):
        # constants
        self.time_step = time_step
        self.time = np.arange(t_initial,t_final+time_step,time_step)
        self.x0 = x0
        self.xDot0 = xDot0
        self.tolerance = tolerance

        # computed state variables
        self.x_hist = np.concatenate((self.x0,np.zeros((len(self.time)-1,(self.x0).size))),axis=0)
        self.xDot_hist = np.concatenate((self.xDot0,np.zeros((len(self.time)-1,(self.x0).size))),axis=0)
        
        # set up state variables
        sim_R['x'] = self.x0
        sim_R['xDot'] = self.xDot0
        sim_R.run()

        # time marching
        for t in range(len(self.time)-1):    
            steps = 0
            x = sim_R['x']
            
            # Newton's method
            while steps < 100:
                # compute xDot
                if t == 0:
                    k0 = 1/self.time_step
                    k1 = -k0
                    xDot = k0*x+k1*self.x_hist[t,:]
                else:
                    k0 = 1.5/self.time_step
                    k1 = -2/self.time_step
                    k2 = 0.5/self.time_step
                    xDot = k0*x+k1*self.x_hist[t,:]+k2*self.x_hist[t-1,:]
                sim_R['xDot'] = xDot
                sim_R.run()
                
                # update residuals
                R = sim_R['R']
                if np.linalg.norm(R) < self.tolerance:
                    break
                
                # compute x
                jacobians_R = sim_R.compute_totals('R',['x','xDot'])
                dRdx = jacobians_R['R','x']
                dRdxDot = jacobians_R['R','xDot']
                delta_x = np.linalg.solve(dRdx+k0*dRdxDot,-R)
                sim_R['x'] = x + delta_x
                sim_R.run()
                
                steps += 1
            
            # record results
            for col in range((self.x0).size):
                self.x_hist[t+1,col] = sim_R['x'][col]
                self.xDot_hist[t+1,col] = xDot[col]
