# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 11:51:31 2022

@author: Zeyu Huang
"""

import numpy as np
from csdl_om import Simulator

class Newton:
    def __init__(self,sim_R,x0,xDot0,t_initial,t_final,time_step,tolerance): #sim_R
        self.time_step = time_step
        self.time = np.arange(t_initial,t_final+time_step,time_step)
        self.x0 = x0
        self.xDot0 = xDot0
        self.tolerance = tolerance
        self.x_hist = np.concatenate((self.x0,np.zeros(((len(self.x0)),len(self.time)-1))),axis=1)
        self.xDot_hist = np.concatenate((self.xDot0,np.zeros(((len(self.xDot0)),len(self.time)-1))),axis=1)
        
        sim = Simulator(sim_R)
        sim['x'] = self.x0
        sim['xDot'] = self.xDot0
        sim.run()
        
        # Time marching
        for t in range(len(self.time)-1):    
            steps = 0
            while steps < 200:
                # Newton's method
                if t == 0:
                    k0 = 1/self.time_step
                    k1 = -k0
                    sim['xDot'] = k0*sim['x']+k1*self.x_hist[:,t]
                    sim.run()
                else:
                    k0 = 1.5/self.time_step
                    k1 = -2/self.time_step
                    k2 = 0.5/self.time_step
                    sim['xDot'] = k0*sim['x']+k1*self.x_hist[:,t]+k2*self.x_hist[:,t-1]
                    sim.run()
                    
                if np.linalg.norm(sim['R']) < self.tolerance:
                    break
                
                jacobians_R = sim.executable.compute_totals('R',['x','xDot']) 
                dRdx = jacobians_R['R','x']
                dRdxDot = jacobians_R['R','xDot']
                delta_x = np.linalg.solve(dRdx+k0*dRdxDot,-sim['R'])
                sim['x'] = sim['x'] + delta_x
                sim.run()

                steps += 1
        
            for row in range(len(self.x0)):
                self.x_hist[row,t+1] = sim['x'][row]
                self.xDot_hist[row,t+1] = sim['xDot'][row]
