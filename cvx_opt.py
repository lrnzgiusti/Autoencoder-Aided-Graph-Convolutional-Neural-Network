#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 12:03:44 2021

@author: ince
"""

import torch
torch.set_default_dtype(torch.float32)
from alegnn.utils.graphLearningTools import Constants, alpha_step, build_duplication_matrix

N = 10
F = 5
mu = 0.8
Z = torch.nn.parameter.Parameter(torch.randn(N,F), requires_grad=False)

N_iter = 2000
alpha = torch.nn.parameter.Parameter(torch.randn( ((N*(N+1))//2), 1))
alpha_v = alpha.data
constants = Constants(10)
alpha_step(alpha_v, constants)
alpha.data.copy_(alpha_v)
#alpha.retain_grad()
#D = build_duplication_matrix(N)

import cvxpy as cp
import numpy as np


## Oracle solution

for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]:# lam = 0.2
    alpha = torch.nn.parameter.Parameter(torch.randn( ((N*(N+1))//2), 1))
    alpha_v = alpha.data
    constants = Constants(10)
    alpha_step(alpha_v, constants)
    alpha.data.copy_(alpha_v)
    L_cvx = cp.Variable([N,N], PSD=True)
    cost = cp.trace(Z.T @ L_cvx @ Z) + lam*cp.sum_squares(cp.vec(L_cvx))
    
    constraints = [(cp.trace(L_cvx) == N), 
                   (L_cvx == cp.transpose(L_cvx)),
                   ((L_cvx-cp.diag(cp.diag(L_cvx))) <= 0),
                   cp.sum(L_cvx, axis=1) == 0
                   ]
    
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()
    
    L = L_cvx.value
    A_cvx= np.diag(np.diag(L)) - L
    #print(A_cvx)
    
    @torch.no_grad()
    def obj(Z, alpha):
        A = (constants.D @ alpha).reshape(N,N)
        L = torch.diag(torch.sum(A, axis=1)) - A
        return torch.trace(((Z.T @ L) @ Z)) + torch.norm(L)**2
    
    for i in range(N_iter):
        #Z = torch.nn.parameter.Parameter(torch.randn(N,F), requires_grad=False)
        A = (constants.D @ alpha).reshape(N,N)
        L = torch.diag(torch.sum(A, axis=1)) - A
        f = torch.trace(((Z.T @ L) @ Z)) + lam*torch.norm(L)**2
        if i % 500 == 0:
            print(f.item())
        f.backward()
        with torch.no_grad():
            alpha -=  mu*alpha.grad
        alpha_v = alpha.data
        
        alpha_step(alpha_v, constants)
        alpha.data.copy_(alpha_v)
        #alpha.grad.zero_()
        #print(alpha)
        
    A = (constants.D @ alpha).reshape(N,N)
    L = torch.diag(torch.sum(A, axis=1)) - A
    print("\n",lam, np.linalg.norm(L.detach().numpy() - L_cvx.value),"\n")
    