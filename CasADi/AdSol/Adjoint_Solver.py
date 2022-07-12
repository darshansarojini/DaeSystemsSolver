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
        self.x_hist = x_hist
        self.xDot_hist = xDot_hist
        self.adj_hist = np.zeros(self.x_hist.shape)
        self.dRdxDot_hist = np.zeros(self.x_hist.shape)
        self.dFdxDot_hist = np.zeros(self.x_hist.shape)
        time_step = time[1]-time[0]
        
        k = model.k     
        x = model.x
        xDot = model.xDot
        F = (1/2)*time_step*k*x[0]
        
        J_dRdx = model.J_dRdx
        J_dRdxDot = model.J_dRdxDot             
        J_dFdx = Function('dFdx',[x,xDot],[jacobian(F,x)])
        J_dFdxDot = Function('dFdx',[x,xDot],[jacobian(F,xDot)])
        
        for t in range(len(self.time)-1,-1,-1):
            x = self.x_hist[:,t]
            xDot = self.xDot_hist[:,t]
            F = (1/2)*time_step*k*x[0]
            dFdx = J_dFdx(x,xDot)
            dFdxDot = J_dFdxDot(x,xDot)
            dRdx = J_dRdx(x,xDot)
            dRdxDot = J_dRdxDot(x,xDot)
            
            if t == 0:
                k0 = 1/time_step
                k1 = -k0
            else:
                k0 = 1.5/time_step
                k1 = -2/time_step
                k2 = 0.5/time_step
            
            A = (dRdx+k0*dRdxDot).T
            
            if t == len(self.time)-1:
                b = -(dFdx+k0*dFdxDot)
            elif t == len(self.time)-2 or t == 0:
                adj_p1 = self.adj_hist[:,t+1]
                dFdxDot_p1 = self.dFdxDot_hist[:,t+1]
                dRdxDot_p1 = self.dRdxDot_hist[:,t+1]
                b = -(dFdx+k0*dFdxDot)-((k1*dRdxDot*adj_p1).T-(k1*dFdxDot_p1).T)
            else:
                adj_p1 = self.adj_hist[:,t+1]
                dFdxDot_p1 = self.dFdxDot_hist[:,t+1]
                dRdxDot_p1 = self.dRdxDot_hist[:,t+1]
                adj_p2 = self.adj_hist[:,t+2]
                dFdxDot_p2 = self.dFdxDot_hist[:,t+2]
                dRdxDot_p2 = self.dRdxDot_hist[:,t+2]
                sum1 = (k1*dRdxDot_p1*adj_p1).transpose()+(k2*dRdxDot_p2*adj_p2).transpose()
                sum2 = (k1*dFdxDot_p1).transpose()+(k2*dFdxDot_p2).transpose()
                b = -(dFdx+k0*dFdxDot)-(sum1-sum2)
                
            adj = solve(A,b)
            
            for row in range(len(adj)):
                self.adj_hist[row,t] = adj[row]
                self.dRdxDot_hist[row,t] = dRdxDot[row]
                self.dFdxDot_hist[row,t] = dFdxDot[row]
        
                    