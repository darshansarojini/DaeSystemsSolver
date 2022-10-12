# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 11:51:31 2022

@author: Zeyu Huang
"""

import numpy as np
from csdl_om import Simulator

class Newton:
    def __init__(self,Model_R,x0,xDot0,t_initial,t_final,time_step,tolerance): #sim_R
        self.time_step = time_step
        self.time = np.arange(t_initial,t_final+time_step,time_step)
        self.x0 = x0
        self.xDot0 = xDot0
        self.tolerance = tolerance
        self.x_hist = np.concatenate((self.x0,np.zeros((len(self.time)-1,(self.x0).size))),axis=0)
        self.xDot_hist = np.concatenate((self.xDot0,np.zeros((len(self.time)-1,(self.x0).size))),axis=0)
        
        sim_R = Simulator(Model_R)
        sim_R['x'] = self.x0
        sim_R['xDot'] = self.xDot0
        sim_R.run()

        # Time marching
        for t in range(len(self.time)-1):    
            steps = 0
            x = sim_R['x']
            
            while steps < 100:
                # Newton's method
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
                 
                R = sim_R['R']
                if np.linalg.norm(R) < self.tolerance:
                    break
                
                jacobians_R = sim_R.executable.compute_totals('R',['x','xDot'])
                dRdx = jacobians_R['R','x']
                dRdxDot = jacobians_R['R','xDot']
                delta_x = np.linalg.solve(dRdx+k0*dRdxDot,-R)
                sim_R['x'] = x + delta_x
                sim_R.run()
                
                steps += 1
        
            for col in range((self.x0).size):
                self.x_hist[t+1,col] = x[col]
                self.xDot_hist[t+1,col] = xDot[col]
