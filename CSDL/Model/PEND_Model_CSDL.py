# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 03:52:08 2022

@author: Vera_
"""

import csdl
from csdl import Model

class SMD_R(Model):
    def initialize(self):        
        self.parameters.declare('l', default=1., types=(float))
        self.parameters.declare('g', default=9.80665, types=(float))
    
    def define(self):         
        x = self.create_input('x', shape=(2,))
        xDot = self.create_input('xDot', shape=(2,))
        
        l = self.create_input('l',val=self.parameters['l'])
        g = self.parameters['g'] 
        self.add_design_variable('l')
        
        R = self.create_output('R', shape=(2,))
        R[0] = g/l*csdl.sin(x[0])+xDot[1]
        R[1] = x[1]-xDot[0]

class SMD_F(Model):
    def initialize(self):
        self.parameters.declare('time_step',default=10**(-3),types=(int,float))
        
        self.parameters.declare('l',default=1,types=(int,float))

    def define(self):      
        x = self.create_input('x', shape=(2,))
        xDot = self.create_input('xDot', shape=(2,))
        
        time_step = self.parameters['time_step']
        
        l = self.create_input('l',val=self.parameters['l'])
        
        self.add_design_variable('l')
        
        F = 1/2*time_step*l*x[0]**2
        self.register_output('F',F)

# from csdl_om import Simulator

# class SMD:
#     def __init__(self):
#         sim_R = Simulator(SMD_R())
#         sim_R['x'] = np.array([1,0])
#         sim_R['xDot'] = np.array([0,0])
#         sim_R.run()
        
#         self.jacobians_R = sim_R.executable.compute_totals('R',['x','xDot'])
#         self.dRdx = self.jacobians_R['R','x']
#         self.dRdxDot = self.jacobians_R['R','xDot']
#         self.R = sim_R['R']
#         self.x = sim_R['x']
#         self.xDot = sim_R['xDot']
        # self.x0 = sim_R['x0']
        # self.xDot0 = sim_R['xDot0']
        
        # sim_F = Simulator(SMD_F())
        # # sim_F['x']=0
        # sim_F.run()
        
        # self.jacobians_F = sim_F.executable.compute_totals('F',['k','m','c','x','xDot'])
        # self.dFdx = self.jacobians_F['F','x']
        # self.dFdxDot = self.jacobians_F['F','xDot']
        # self.time_step = sim_F['time_step']