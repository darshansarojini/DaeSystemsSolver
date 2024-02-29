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

import numpy as np

"""
Residuals
"""
class DOUBLE_PEND_R(Model):
    # initialize design parameters
    def initialize(self):        
        self.parameters.declare('l1', default=1., types=(float))
        self.parameters.declare('l2', default=1., types=(float))
        # self.parameters.declare('g', default=9.80665, types=(float))
        self.parameters.declare('g', default=10., types=(float))
        self.parameters.declare('b', default=0.5, types=(float))
        self.parameters.declare('m1', default=1., types=(float))
        self.parameters.declare('m2', default=1., types=(float))
    
    # define function of interest
    def define(self):
        # state variables
        x = self.create_input('x', shape=(20,))
        xDot = self.create_input('xDot', shape=(20,))
        
        # design variables
        l1 = self.create_input('l1',val=self.parameters['l1'])
        l2 = self.create_input('l2',val=self.parameters['l2'])
        b = self.create_input('b',val=self.parameters['b'])
        m1 = self.create_input('m1',val=self.parameters['m1'])
        m2 = self.create_input('m2',val=self.parameters['m2'])
        g = self.parameters['g'] 
        self.add_design_variable('l1')
        self.add_design_variable('l2')
        self.add_design_variable('b')
        self.add_design_variable('m1')
        self.add_design_variable('m2')
        
        # residual
        R = self.create_output('R', shape=(20,), val=np.zeros((20,)))

        g_ext = self.create_output('g_ext', shape=(6,), val=np.zeros((6,)))
        g_ext[1] = -m1*g
        g_ext[4] = -m2*g
        # g_ext = np.array([0., -m1*g, 0.,  0., -m2*g, 0.])
        # g_ext = self.create_input('g_ext', val=g_ext)
        one = self.create_input('one', val=1., shape=(1,1))
        # B = self.create_output('B_T', shape=(6,4), val=np.zeros((6,4)))
        # B[0,0] = one*1
        # B[2,0] = csdl.reshape(-l1*csdl.cos(x[2]), (1,1))
        # B[1,1] = one*1
        # B[2,1] = csdl.reshape(-l1*csdl.sin(x[2]), (1,1))
        # B[0,2] = -one*1
        # B[3,2] = one*1
        # B[5,2] = csdl.reshape(-l2*csdl.cos(x[5]), (1,1))
        # B[3,3] = one*1
        # B[1,3] = -one*1
        # B[4,3] = one*1
        # B[5,3] = csdl.reshape(-l2*csdl.sin(x[5]), (1,1))

        B = self.create_output('B', val=np.zeros((4,6)))
        B[0,0] = one*1
        B[0,2] = csdl.reshape(-l1*csdl.cos(x[2]), (1,1))
        B[1,1] = one*1
        B[1,2] = csdl.reshape(-l1*csdl.sin(x[2]), (1,1))
        B[2,0] = -one*1
        B[2,3] = one*1
        B[2,5] = csdl.reshape(-l2*csdl.cos(x[5]), (1,1))
        B[3,1] = -one*1
        B[3,4] = one*1
        B[3,5] = csdl.reshape(-l2*csdl.sin(x[5]), (1,1))
        B_T = csdl.transpose(B)

        # B = np.array([
        #     [1., 0., -l1*csdl.cos(x[2]), 0., 0., 0.],
        #     [0., 1., -l1*csdl.sin(x[2]), 0., 0., 0.],
        #     [-1., 0., 0., 1., 0., -l2*csdl.cos(x[5])],
        #     [0., -1., 0., 0., 1., -l2*csdl.sin(x[5])]
        # ]).T
        # B = self.create_input('B_T', val=B.T)

        M = self.create_output('M', shape=(6,6), val=np.zeros((6,6)))
        M[0,0] = csdl.reshape(m1, (1,1))
        M[1,1] = csdl.reshape(m1, (1,1))
        M[3,3] = csdl.reshape(m2, (1,1))
        M[4,4] = csdl.reshape(m2, (1,1))
        # M = np.array([
        #     [m1, 0., 0., 0., 0., 0.],
        #     [0., m1, 0., 0., 0., 0.],
        #     [0., 0., 0., 0., 0., 0.],
        #     [0., 0., 0., m2, 0., 0.],
        #     [0., 0., 0., 0., m2, 0.],
        #     [0., 0., 0., 0., 0., 0.]
        # ])
        # M = self.create_input('M', val=M)

        R[0:6] = g_ext - csdl.matvec(B_T, x[6:10]) - csdl.matvec(M, xDot[10:16])
        R[6] = x[0] - l1*csdl.sin(x[2])
        R[7] = x[1] + l1*csdl.cos(x[2])
        R[8] = x[3] - x[0] - l2*csdl.sin(x[5])
        R[9] = x[4] - x[1] + l2*csdl.cos(x[5])
        R[10:20] = xDot[0:10] - x[10:20]
        # R[0] = g/l*csdl.sin(x[0])+b/m*x[1]+xDot[1]
        # R[1] = x[1]-xDot[0]

"""
Function of Interest
"""
class DOUBLE_PEND_F(Model):
    # define design parameters
    def initialize(self):
        self.parameters.declare('time_step',default=10**(-3),types=(int,float))
        
        self.parameters.declare('l1', default=1., types=(float))
        self.parameters.declare('l2', default=1., types=(float))
        self.parameters.declare('g', default=9.80665, types=(float))
        self.parameters.declare('b', default=0.5, types=(float))
        self.parameters.declare('m1', default=1., types=(float))
        self.parameters.declare('m2', default=1., types=(float))

    # define function of interest (objective function for adjoint method)
    def define(self):
        # state variables
        x = self.create_input('x', shape=(20,))
        xDot = self.create_input('xDot', shape=(20,))
        
        # constants
        time_step = self.parameters['time_step']
        g = self.parameters['g'] 
        
        # design variables
        l1 = self.create_input('l1',val=self.parameters['l1'])
        l2 = self.create_input('l2',val=self.parameters['l2'])
        m1 = self.create_input('m1',val=self.parameters['m1'])
        m2 = self.create_input('m2',val=self.parameters['m2'])
        b = self.create_input('b',val=self.parameters['b'])
        self.add_design_variable('l1')
        self.add_design_variable('l2')
        self.add_design_variable('b')
        self.add_design_variable('m1')
        self.add_design_variable('m2')
        
        # kinetic energy of the pendulum
        F = 1/2*m1*(xDot[0])**2 + 1/2*m1*(xDot[1])**2 + 1/2*m2*(xDot[3])**2 + 1/2*m2*(xDot[4])**2

        self.register_output('F',F)

