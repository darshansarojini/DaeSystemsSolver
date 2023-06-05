# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 03:52:08 2022

@author: Zeyu Huang

This is an example of spring-mass-damper system in purpose of demonstrating 
how to construct CSDL models of DAE systems. This script contains two classes
corresponding to residual and function of interest.
"""

from csdl import Model

"""
Residuals
"""
class SMD_R(Model):
    # initialize design parameters
    def initialize(self):        
        self.parameters.declare('c', default=0.2, types=(int,float))
        self.parameters.declare('m', default=2.5, types=(int,float))
        self.parameters.declare('k', default=5.0, types=(int,float))
    
    # define DAE system
    def define(self):
        # state variables
        x = self.create_input('x', shape=(2,))
        xDot = self.create_input('xDot', shape=(2,))
        
        # design variables
        c = self.create_input('c',val=self.parameters['c'])
        m = self.create_input('m',val=self.parameters['m'])
        k = self.create_input('k',val=self.parameters['k'])
        self.add_design_variable('c')
        self.add_design_variable('m')
        self.add_design_variable('k')
        
        # residuals
        R = self.create_output('R', shape=(2,))
        R[0] = m*xDot[1]+c*x[1]+k*x[0]
        R[1] = x[1]-xDot[0]


"""
Function of Interest
"""
class SMD_F(Model):
    # initialize design parameters
    def initialize(self):
        self.parameters.declare('time_step',default=10**(-3),types=(int,float))
        
        self.parameters.declare('c',default=0.2,types=(int,float))
        self.parameters.declare('m',default=2.5,types=(int,float))
        self.parameters.declare('k',default=5.0,types=(int,float))

    # define function of interest
    def define(self):
        # state variables
        x = self.create_input('x', shape=(2,))
        xDot = self.create_input('xDot', shape=(2,))
        
        # constant
        time_step = self.parameters['time_step']
        
        # design variables
        c = self.create_input('c',val=self.parameters['c'])
        m = self.create_input('m',val=self.parameters['m'])
        k = self.create_input('k',val=self.parameters['k'])
        self.add_design_variable('c')
        self.add_design_variable('m')
        self.add_design_variable('k')
        
        # elastic energy of the spring
        F = 1/2*time_step*k*x[0]**2
        self.register_output('F',F)

