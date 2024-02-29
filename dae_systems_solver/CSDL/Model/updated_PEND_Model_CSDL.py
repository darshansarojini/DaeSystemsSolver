# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 03:52:08 2022

@author: Zeyu Huang

This is an example of pendulum system in purpose of demonstrating how to 
construct CSDL models of DAE systems. This script contains two classes 
corresponding to residual and function of interest.
"""

import csdl
from csdl import Model

"""
Residuals
"""
class DOUBLE_PEND_R(Model):
    # initialize design parameters
    def initialize(self):        
        self.parameters.declare('l', default=1., types=(float))
        self.parameters.declare('g', default=9.80665, types=(float))
        self.parameters.declare('b', default=0.5, types=(float))
        self.parameters.declare('m', default=1., types=(float))
    
    # define function of interest
    def define(self):
        # state variables
        x = self.create_input('x', shape=(2,))
        xDot = self.create_input('xDot', shape=(2,))
        
        # design variables
        l = self.create_input('l',val=self.parameters['l'])
        b = self.create_input('b',val=self.parameters['b'])
        m = self.create_input('m',val=self.parameters['m'])
        g = self.parameters['g'] 
        self.add_design_variable('l')
        self.add_design_variable('b')
        self.add_design_variable('m')
        
        # residual
        R = self.create_output('R', shape=(2,))
        R[0] = g/l*csdl.sin(x[0])+b/m*x[1]+xDot[1]
        R[1] = x[1]-xDot[0]

"""
Function of Interest
"""
class DOUBLE_PEND_F(Model):
    # define design parameters
    def initialize(self):
        self.parameters.declare('time_step',default=10**(-3),types=(int,float))
        
        self.parameters.declare('l', default=1., types=(float))
        self.parameters.declare('g', default=9.80665, types=(float))
        self.parameters.declare('b', default=0.5, types=(float))
        self.parameters.declare('m', default=1., types=(float))

    # define function of interest
    def define(self):
        # state variables
        x = self.create_input('x', shape=(2,))
        xDot = self.create_input('xDot', shape=(2,))
        
        # constants
        time_step = self.parameters['time_step']
        g = self.parameters['g'] 
        
        # design variables
        l = self.create_input('l',val=self.parameters['l'])
        m = self.create_input('m',val=self.parameters['m'])
        b = self.create_input('b',val=self.parameters['b'])
        self.add_design_variable('l')
        self.add_design_variable('b')
        self.add_design_variable('m')
        
        # kinetic energy of the pendulum
        F = 1/2*time_step*m*(x[1]*l)**2
        self.register_output('F',F)

