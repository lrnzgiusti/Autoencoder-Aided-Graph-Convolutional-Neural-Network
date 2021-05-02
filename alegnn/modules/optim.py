#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 22:14:54 2021

@author: ince
"""

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from alegnn.utils.graphLearningTools import alpha_step

class MultiGraphLearningOptimizer(Optimizer):
    
    def __init__(self, named_parameters, constants,
                 lr=0.03, 
                 weight_decay=0.0,
                 betas=(0.9, 0.999), 
                 momentum=0.0):
        
        def get_params():
            """
            Get the iterator over the parameters you need
            We have two kind of parameters:
                - Filters Parameters
                - Graphs Parameters

            Parameters
            ----------
            

            Returns
            ------
            filters_params : torch.tensor
                Parameters of the filters and the MLP.
            graphs_params : torch.tensor
                Parameters of the graph

            """
            filters_params = []
            graphs_params = []
            for name, param in named_parameters:
                if 'alpha' not in name: 
                    filters_params.append(param)
                else: 
                    graphs_params.append(param)
                    
            return filters_params, graphs_params
                    
        
        
        #Get the filters parameters h_lk^fg
        self.filters_params, self.graphs_params = get_params()
        
        params = self.filters_params + self.graphs_params
        
        defaults = dict(lr=lr, betas=betas, eps=1e-8,
                        weight_decay=weight_decay, momentum=momentum)
        super(MultiGraphLearningOptimizer, self).__init__(params, defaults)
        
        self.filters_opt = optim.Adam(self.filters_params,
                                     lr=lr,
                                     betas=betas,
                                     weight_decay=weight_decay)
        
        self.graphs_opt = optim.SGD(self.graphs_params, 
                                    lr=lr,
                                    weight_decay=weight_decay,
                                    momentum=momentum)
        self.constants = constants 
        
    @torch.no_grad()
    def step(self, closure=None):
        self.filters_opt.step()
        self.graphs_opt.step()
        for i, param in enumerate(self.graphs_params):
            alpha_step(param, self.constants[i])
        