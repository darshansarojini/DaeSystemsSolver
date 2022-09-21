# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 03:52:08 2022

@author: Vera_
"""

from csdl import Model,ScipyKrylov,NewtonSolver
import csdl
import numpy as np

class SMD_R(Model):
    def initialize(self):
        self.parameters.declare('x0', default=np.array([1,0]), types=np.ndarray)
        self.parameters.declare('xDot0', default=np.array([0,0]), types=np.ndarray)
        
        self.parameters.declare('c', default=0.2, types=(int,float))
        self.parameters.declare('m', default=2.5, types=(int,float))
        self.parameters.declare('k', default=5.0, types=(int,float))
    
    def define(self):         
        x = self.create_input('x', shape=(2,),val=self.parameters['x0'])
        xDot = self.create_input('xDot', shape=(2,),val=self.parameters['xDot0'])
        
        c = self.create_input('c',val=self.parameters['c'])
        m = self.create_input('m',val=self.parameters['m'])
        k = self.create_input('k',val=self.parameters['k'])
        
        self.add_design_variable('c')
        self.add_design_variable('m')
        self.add_design_variable('k')
        
        R = self.create_output('R', shape=(2,))
        R[0] = m*xDot[1]+c*x[1]+k*x[0]
        R[1] = x[1]-xDot[0]

class SMD_F(Model):
    def initialize(self):
        self.parameters.declare('x0',default=np.array([1,0]),types=np.ndarray)
        self.parameters.declare('xDot0',default=np.array([0,0]),types=np.ndarray)
        
        self.parameters.declare('time_step',default=10**(-3),types=(int,float))
        
        self.parameters.declare('c',default=0.2,types=(int,float))
        self.parameters.declare('m',default=2.5,types=(int,float))
        self.parameters.declare('k',default=5.0,types=(int,float))

    def define(self):      
        x = self.create_input('x', shape=(2,),val=self.parameters['x0'])
        xDot = self.create_input('xDot', shape=(2,),val=self.parameters['xDot0'])
        
        time_step = self.parameters['time_step']
        
        c = self.create_input('c',val=self.parameters['c'])
        m = self.create_input('m',val=self.parameters['m'])
        k = self.create_input('k',val=self.parameters['k'])
        
        self.add_design_variable('c')
        self.add_design_variable('m')
        self.add_design_variable('k')
        
        F = 1/2*time_step*k*x[0]**2
        self.register_output('F',F)

from csdl_om import Simulator

sim_R = Simulator(SMD_R())
sim_R.run()

jacobians_R = sim_R.executable.compute_totals('R',['k','m','c','x','xDot'])
dRdx = jacobians_R['R','x']
dRdxDot = jacobians_R['R','xDot']
print(dRdx)
print(dRdxDot)

sim_F = Simulator(SMD_F())
sim_F.run()

jacobians_F = sim_F.executable.compute_totals('F',['k','m','c','x','xDot'])
dFdx = jacobians_F['F','x']
dFdxDot = jacobians_F['F','xDot']
print(dFdx)
print(dFdxDot)