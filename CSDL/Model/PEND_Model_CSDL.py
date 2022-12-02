# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 03:52:08 2022

@author: Zeyu Huang
"""

import csdl
from csdl import Model

class PEND_R(Model):
    def initialize(self):        
        self.parameters.declare('l', default=1., types=(float))
        self.parameters.declare('g', default=9.80665, types=(float))
        self.parameters.declare('b', default=0.5, types=(float))
        self.parameters.declare('m', default=1., types=(float))
    
    def define(self):         
        x = self.create_input('x', shape=(2,))
        xDot = self.create_input('xDot', shape=(2,))
        
        l = self.create_input('l',val=self.parameters['l'])
        b = self.create_input('b',val=self.parameters['b'])
        m = self.create_input('m',val=self.parameters['m'])
        g = self.parameters['g'] 
        self.add_design_variable('l')
        self.add_design_variable('b')
        self.add_design_variable('m')
        
        R = self.create_output('R', shape=(2,))
        R[0] = g/l*csdl.sin(x[0])+b/m*x[1]+xDot[1]
        R[1] = x[1]-xDot[0]

class PEND_F(Model):
    def initialize(self):
        self.parameters.declare('time_step',default=10**(-3),types=(int,float))
        
        self.parameters.declare('l', default=1., types=(float))
        self.parameters.declare('g', default=9.80665, types=(float))
        self.parameters.declare('b', default=0.5, types=(float))
        self.parameters.declare('m', default=1., types=(float))

    def define(self):      
        x = self.create_input('x', shape=(2,))
        xDot = self.create_input('xDot', shape=(2,))
        
        time_step = self.parameters['time_step']
        
        l = self.create_input('l',val=self.parameters['l'])
        m = self.create_input('m',val=self.parameters['m'])
        b = self.create_input('b',val=self.parameters['b'])
        
        g = self.parameters['g'] 
        self.add_design_variable('l')
        self.add_design_variable('b')
        self.add_design_variable('m')
        
        F = 1/2*m*(x[1]*l)**2
        self.register_output('F',F)

