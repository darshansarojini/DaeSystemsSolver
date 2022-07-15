# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 12:13:16 2022

@author: Zeyu Huang
"""

from casadi import *
import numpy as np

class Adjoint:
    def __init__(self,model,time,x_hist,xDot_hist):
        self.time = time
        self.time_step = time[1]-time[0]
        self.x_hist = x_hist
        self.xDot_hist = xDot_hist
        self.adj_hist = np.zeros(self.x_hist.shape)
        self.dRdxDot_adj_hist = np.zeros(self.x_hist.shape)
        self.dFdxDot_hist = np.zeros(self.x_hist.shape)

        J_dRdx = model.J_dRdx
        J_dRdxDot = model.J_dRdxDot             
        J_dFdx = model.J_dFdx
        J_dFdxDot = model.J_dFdxDot        
        mu = model.mu
        
        for t in range(len(self.time)-1,-1,-1):
            x = self.x_hist[:,t]
            xDot = self.xDot_hist[:,t]
            dFdx = J_dFdx(x,xDot,mu).T
            dFdxDot = J_dFdxDot(x,xDot,mu).T
            dRdx = J_dRdx(x,xDot,mu)
            dRdxDot = J_dRdxDot(x,xDot,mu)
            
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
                dFdxDot_p1 = self.dFdxDot_hist[:,t+1]
                dRdxDot_adj_p1 = self.dRdxDot_adj_hist[:,t+1]
                b = -(dFdx+k0*dFdxDot)-(k1*dRdxDot_adj_p1-k1*dFdxDot_p1)
            else:
                dRdxDot_adj_p1 = self.dRdxDot_adj_hist[:,t+1]
                dRdxDot_adj_p2 = self.dRdxDot_adj_hist[:,t+2]
                dFdxDot_p1 = self.dFdxDot_hist[:,t+1]
                dFdxDot_p2 = self.dFdxDot_hist[:,t+2]                
                sum1 = (k1*dRdxDot_adj_p1)+(k2*dRdxDot_adj_p2)
                sum2 = (k1*dFdxDot_p1)+(k2*dFdxDot_p2)
                b = -(dFdx+k0*dFdxDot)-(sum1-sum2)
                
            A = (dRdx+k0*dRdxDot).T    
            adj = solve(A,b)
            
            for row in range(len(x_hist[:,0])):
                self.adj_hist[row,t] = adj[row]
                dRdxDot_adj = dRdxDot.T @ adj
                self.dRdxDot_adj_hist[row,t] = dRdxDot_adj[row]
                self.dFdxDot_hist[row,t] = dFdxDot[row]

    def final_dfdk(self,model):
        mu = model.mu
        adj = SX.sym('adj',2,1)
        dfdk = 0
        for t in range(len(self.time)-1):
            x = self.x_hist[:,t]
            xDot = self.xDot_hist[:,t]
            adj = self.adj_hist[:,t]
            dFdmu = model.J_dFdmu(x,xDot,model.mu)
            dRdmu = model.J_dRdmu(x,xDot,model.mu)
            dfdk = dfdk + dFdmu[2] + dot(adj.T,dRdmu[:,2])
        return dfdk