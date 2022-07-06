# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 21:53:39 2022

@author: Zeyu Huang
"""

from casadi import *
import numpy as np

class Newton:
    def __init__(self,model,t_initial,t_final,time_step,tolerance):
        self.time = np.arange(t_initial,t_final+time_step,time_step)
        self.x0 = model.x0
        self.xDot0 = model.xDot0
        self.tolerance = tolerance
        self.x_hist = np.concatenate((self.x0,np.zeros(((len(self.x0)),len(self.time)-1))),axis=1)
        self.xDot_hist = np.concatenate((self.xDot0,np.zeros(((len(self.xDot0)),len(self.time)-1))),axis=1)
        
        x = model.x
        xDot = model.xDot
        
        J_dRdx = model.J_dRdx
        J_dRdxDot = model.J_dRdxDot
        Residuals = model.Residuals
        
        # Time marching
        for t in range(len(self.time)-1):    
            x = self.x_hist[:,t]
            steps = 0
            while steps < 200:
                # Newton's method
                if t == 0:
                    k0 = 1/time_step
                    k1 = -k0
                    xDot = k0*x+k1*self.x_hist[:,t]
                else:
                    k0 = 1.5/time_step
                    k1 = -2/time_step
                    k2 = 0.5/time_step
                    xDot = k0*x+k1*self.x_hist[:,t]+k2*self.x_hist[:,t-1]             
                r = Residuals(x,xDot)
                if norm_1(r) < self.tolerance:
                    break
                drdx = J_dRdx(x,xDot)
                drdxDot = J_dRdxDot(x,xDot)
                delta_x = solve(drdx+k0*drdxDot,-r)
                x = x + delta_x
                steps += 1
        
            for row in range(len(self.x0)):
                self.x_hist[row,t+1] = x[row]
                self.xDot_hist[row,t+1] = xDot[row]