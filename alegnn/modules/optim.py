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
    """
    MultiGraphLearningOptimizer: Optimizer for the multigraph learning task
    
    We perform a single optimization step for the network's weights and then
    optimize the hidden graphs
    
    TODO: set a learning rate different for graph and parameters
    
    The weights are optimized using adam
    The graphs are optimized using SGD 
    """
    def __init__(self, named_parameters, constants,
                 lr=0.03, 
                 weight_decay=0.0,
                 betas=(0.9, 0.999), 
                 momentum=0.0):
        
        def get_params():
            """
            Get the network parameters as two independent collections
            We have two kind of parameters:
                - Filters Parameters
                - Graphs Parameters

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
                #graph parameters does not contain alpha in the name
                if 'alpha' not in name: 
                    filters_params.append(param)
                else: 
                    graphs_params.append(param)
                    
            return filters_params, graphs_params
                    
        
        
        #Get the filters parameters h_lk^fg and alpha_l
        self.filters_params, self.graphs_params = get_params()
        
        #for representation purposes
        params = self.filters_params + self.graphs_params
        
        defaults = dict(lr=lr, betas=betas, eps=1e-8,
                        weight_decay=weight_decay, momentum=momentum)
        super(MultiGraphLearningOptimizer, self).__init__(params, defaults)
        
        #optimization algorithm for the network's parameters
        self.filters_opt = optim.Adam(self.filters_params,
                                     lr=lr,
                                     betas=betas,
                                     weight_decay=weight_decay)
        
        #optimization algorithm for the hidden graphs
        self.graphs_opt = optim.SGD(self.graphs_params, 
                                    lr=lr,
                                    weight_decay=weight_decay,
                                    momentum=momentum)
        #alpha step is performed using the constants object
        self.constants = constants 
        
    @torch.no_grad()
    def step(self, closure=None):
        self.filters_opt.step() #optimize parameters
        #here the computational graph would be freed, in the loss call we 
        #have to specify loss.backward(retain_graph=True) or find a way
        #to avoid the release of the computationals graph variables
        self.graphs_opt.step() 
        #project alpha onto the feasible set
        for i, param in enumerate(self.graphs_params):
            alpha_step(param, self.constants[i])
        