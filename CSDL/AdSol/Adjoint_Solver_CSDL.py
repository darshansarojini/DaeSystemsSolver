# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 10:12:51 2022

@author: Zeyu Huang
"""

import numpy as np
from csdl_om import Simulator

class Adjoint:
    def __init__(self,Model_R,Model_F,time,x_hist,xDot_hist,mu):
        self.time = time
        self.time_step = time[1]-time[0]
        self.x_hist = x_hist
        self.xDot_hist = xDot_hist
        self.adj_hist = np.zeros(self.x_hist.shape)
        self.dRdxDot_adj_hist = np.zeros(self.x_hist.shape)
        self.dFdxDot_hist = np.zeros(self.x_hist.shape)
        self.mu = mu

        self.sim_R = Simulator(Model_R)
        self.sim_F = Simulator(Model_F)
        
        for t in range(len(self.time)-1,-1,-1):
            self.sim_R['x'] = self.x_hist[t,:]
            self.sim_R['xDot'] = self.xDot_hist[t,:]
            self.sim_R.run()
            self.sim_F['x'] = self.x_hist[t,:]
            self.sim_F['xDot'] = self.xDot_hist[t,:]
            self.sim_F.run()
            
            jacobians_R = self.sim_R.executable.compute_totals('R',['x','xDot']) 
            dRdx = jacobians_R['R','x']
            dRdxDot = jacobians_R['R','xDot']
            
            jacobians_F = self.sim_F.executable.compute_totals('F',['x','xDot'])
            dFdx = jacobians_F['F','x']
            dFdxDot = jacobians_F['F','xDot']
            
            if t == 0:
                k0 = 1/self.time_step
                k1 = -k0
            else:
                k0 = 1.5/self.time_step
                k1 = -2/self.time_step
                k2 = 0.5/self.time_step
            
            if t == len(self.time)-1:
                b = -(dFdx+k0*dFdxDot)
            elif t == len(self.time)-2 or t == 0:
                dFdxDot_p1 = self.dFdxDot_hist[t+1,:]
                dRdxDot_adj_p1 = self.dRdxDot_adj_hist[t+1,:]
                b = -(dFdx+k0*dFdxDot)-(k1*dRdxDot_adj_p1-k1*dFdxDot_p1)
            else:
                dRdxDot_adj_p1 = self.dRdxDot_adj_hist[t+1,:]
                dRdxDot_adj_p2 = self.dRdxDot_adj_hist[t+2,:]
                dFdxDot_p1 = self.dFdxDot_hist[t+1,:]
                dFdxDot_p2 = self.dFdxDot_hist[t+2,:]                
                sum1 = (k1*dRdxDot_adj_p1) + (k2*dRdxDot_adj_p2)
                sum2 = (k1*dFdxDot_p1) + (k2*dFdxDot_p2)
                b = - (dFdx+k0*dFdxDot) - (sum1-sum2)
                
            A = (dRdx+k0*dRdxDot).T
            adj = np.linalg.solve(A,b.T)
            
            for col in range((x_hist[0,:]).size):
                self.adj_hist[t,col] = adj[col]
                dRdxDot_adj = np.matmul(dRdxDot.T,adj)
                self.dRdxDot_adj_hist[t,col] = dRdxDot_adj[col]
                self.dFdxDot_hist[t,col] = dFdxDot[0,col]

    # nu should be in the form of list
    def final_grad(self):
        mu = self.mu
        final_grad = np.zeros(len(mu))
        for t in range(len(self.time)-1):
            self.sim_R['x'] = self.x_hist[t,:]
            self.sim_R['xDot'] = self.xDot_hist[t,:]
            self.sim_R.run()
            self.sim_F['x'] = self.x_hist[t,:]
            self.sim_F['xDot'] = self.xDot_hist[t,:]
            self.sim_F.run()

            adj = self.adj_hist[t,:]
            jacobians_R = self.sim_R.executable.compute_totals('R',mu) 
            jacobians_F = self.sim_F.executable.compute_totals('F',mu)

            for i in range(len(mu)):
                dRdmu = jacobians_R['R',mu[i]]
                dFdmu = jacobians_F['F',mu[i]]
                final_grad[i] = final_grad[i] + dFdmu + np.dot(adj,dRdmu)
        return final_grad
