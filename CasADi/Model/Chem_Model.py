# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 10:30:09 2022

@author: Vera_
"""

from casadi import *

class ChemReaction:
    def __init__(self,A,B,C,x0,xDot0,time_step):
        self.mu = SX.sym('mu',3,1)
        self.x0 = x0
        self.xDot0 = xDot0
        self.R = SX.sym('R',3,1)
        self.x = SX.sym('x',3,1)
        self.xDot = SX.sym('xDot',3,1)
        self.time_step = time_step
        
        self.R[0] = -self.mu[0]*self.x[0]+self.mu[2]*self.x[1]*self.x[2]-self.xDot[0]
        self.R[1] = self.mu[0]*self.x[0]-self.mu[2]*self.x[1]*self.x[2]-self.mu[1]*self.x[1]*self.x[1]-self.xDot[1]
        self.R[2] = self.mu[1]*self.x[1]*self.x[1]-self.xDot[2]

        self.Residuals = Function('R',[self.x,self.xDot,self.mu],[self.R])
        self.J_dRdx = Function('dRdx',[self.x,self.xDot,self.mu],[jacobian(self.R,self.x)])
        self.J_dRdxDot = Function('dRdxDot',[self.x,self.xDot,self.mu],[jacobian(self.R,self.xDot)])
        self.J_dRdmu = Function('dFdx',[self.x,self.xDot,self.mu],[jacobian(self.R,self.mu)])
        
        self.mu[0] = A
        self.mu[1] = B
        self.mu[2] = C