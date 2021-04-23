#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 21:59:54 2021

@author: renn
"""


import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.cl1 = nn.Linear(25, 60)
        self.cl2 = nn.Linear(60, 16)
        self.fc1 = nn.Linear(16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.seq = nn.Sequential(self.cl1,
                                 nn.ReLU(),
                                 self.cl2,
                                 nn.ReLU(),
                                 self.fc1,
                                 nn.ReLU(),
                                 self.fc2,
                                 nn.ReLU(),
                                 self.fc3,
                                 nn.ReLU())
        
    def forward(self, x):
        x = self.seq(x)
        x = F.log_softmax(x, dim=1)
        return x


activation = {}
def get_activation(name):
    def hook(model, input, output):
        print(model)
        activation[name] = output.detach()
    return hook


model = MyModel()
model.seq[4].register_forward_hook(get_activation('fc1'))
model.seq[5].register_forward_hook(get_activation('activ_fc1'))
x = torch.randn(1, 25)
output = model(x)
print(activation['fc1'])
print(activation['activ_fc1'])


"""


output = model(x)
print(activation['fc1'])
print(activation['activ_fc1'])
"""
