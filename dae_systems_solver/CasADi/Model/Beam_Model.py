# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 10:15:49 2022

@author: Vera_
"""

from casadi import *

class SpringMassDamper:
    def __init__(self,c,m,k,r0,rDot0,theta0,omega0,F0,M0,time_step):
        self.mu = SX.sym('mu',3,1)
        self.x0 = x0
        self.xDot0 = xDot0
        self.R = SX.sym('R',2,1)
        self.x = SX.sym('x',2,1)
        self.xDot = SX.sym('xDot',2,1)
        self.time_step = time_step
        
        self.R[0] = self.mu[0]*self.xDot[1]+self.mu[1]*self.x[1]+self.mu[2]*self.x[0]
        self.R[1] = self.x[1]-self.xDot[0]

        self.Residuals = Function('R',[self.x,self.xDot,self.mu],[self.R])
        self.J_dRdx = Function('dRdx',[self.x,self.xDot,self.mu],[jacobian(self.R,self.x)])
        self.J_dRdxDot = Function('dRdxDot',[self.x,self.xDot,self.mu],[jacobian(self.R,self.xDot)])
        self.J_dRdmu = Function('dFdx',[self.x,self.xDot,self.mu],[jacobian(self.R,self.mu)])
        
        self.F = (1/2)*self.time_step*self.mu[2]*self.x[0]**2
        self.J_dFdx = Function('dFdx',[self.x,self.xDot,self.mu],[jacobian(self.F,self.x)])
        self.J_dFdxDot = Function('dFdx',[self.x,self.xDot,self.mu],[jacobian(self.F,self.xDot)])        
        self.J_dFdmu = Function('dFdx',[self.x,self.xDot,self.mu],[jacobian(self.F,self.mu)])
        
        self.mu[0] = m
        self.mu[1] = c
        self.mu[2] = k