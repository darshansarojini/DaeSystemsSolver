# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 03:52:08 2022

@author: Zeyu Huang
"""

import csdl
from csdl import Model

class RLC_R(Model):
    def initialize(self):        
        self.parameters.declare('RE', default=1.1, types=(float))
        self.parameters.declare('L', default=1.6, types=(float))
        self.parameters.declare('C', default=0.8, types=(float))
        self.parameters.declare('V', default=2.4, types=(float))
    
    def define(self):         
        x = self.create_input('x', shape=(4,))
        xDot = self.create_input('xDot', shape=(4,))
        
        RE = self.create_input('RE',val=self.parameters['RE'])
        L = self.create_input('L',val=self.parameters['L'])
        C = self.create_input('C',val=self.parameters['C'])
        V = self.create_input('V',val=self.parameters['V'])
        self.add_design_variable('RE')
        self.add_design_variable('L')
        self.add_design_variable('C')
        self.add_design_variable('V')
        
        R = self.create_output('R', shape=(4,))
        R[0] = x[1]-L*xDot[0]
        R[1] = 1/C*x[0]-xDot[2]
        R[2] = x[3]-RE*x[0]
        R[3] = x[1]+x[2]+x[3]-V

class RLC_F(Model):
    def initialize(self):
        self.parameters.declare('time_step',default=10**(-3),types=(int,float))

        self.parameters.declare('RE', default=1.1, types=(float))
        self.parameters.declare('L', default=1.6, types=(float))
        self.parameters.declare('C', default=0.8, types=(float))
        self.parameters.declare('V', default=2.4, types=(float))

    def define(self):      
        x = self.create_input('x', shape=(4,))
        xDot = self.create_input('xDot', shape=(4,))

        time_step = self.parameters['time_step']

        RE = self.create_input('RE',val=self.parameters['RE'])
        L = self.create_input('L',val=self.parameters['L'])
        C = self.create_input('C',val=self.parameters['C'])
        V = self.create_input('V',val=self.parameters['V'])
        self.add_design_variable('RE')
        self.add_design_variable('L')
        self.add_design_variable('C')
        self.add_design_variable('V')
        
        F = time_step*V*x[0]
        self.register_output('F',F)

