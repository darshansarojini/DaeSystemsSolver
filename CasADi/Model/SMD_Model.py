# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 23:24:30 2022

@author: Zeyu Huang
"""

from casadi import *

class SpringMassDamper:
    def __init__(self,c,m,k,x0,xDot0):
        self.c = c
        self.m = m
        self.k = k
        self.x0 = x0
        self.xDot0 = xDot0
        self.R = SX.sym('R',2,1)
        self.x = SX.sym('x',2,1)
        self.xDot = SX.sym('xDot',2,1)
        self.R[0] = m*self.xDot[1]+c*self.x[1]+k*self.x[0]
        self.R[1] = self.x[1]-self.xDot[0]
        dRdx = jacobian(self.R,self.x)
        dRdxDot = jacobian(self.R,self.xDot)
        self.Residuals = Function('R',[self.x,self.xDot],[self.R])
        self.J_dRdx = Function('dRdx',[self.x,self.xDot],[dRdx])
        self.J_dRdxDot = Function('dRdxDot',[self.x,self.xDot],[dRdxDot])