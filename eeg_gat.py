#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 17:23:17 2021

@author: ince
"""




import sys
sys.path.append(".")
sys.path.append("..")

import torch
import torch.nn as nn
import pytorch_lightning as pl
spmm = torch.sparse.mm

import pickle


import numpy as np
import alegnn.utils.graphTools as graphTools
import alegnn.utils.graphML as gml
import alegnn.modules.architectures as archit
from alegnn.modules.loss import MultiGraphLearningLoss



    

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path = r"/Users/ince/Desktop/base/EEGNet/Popularity-Index-Prediction/" ):
        X_y = pickle.load(open(data_path+"X_y.pkl", "rb"))
        eeg = pickle.load(open(data_path+"eeg_new.pkl", "rb"))
        self.eeg = torch.from_numpy(eeg).double()
        self.y = torch.from_numpy(X_y['y']).double()
        
    def __getitem__(self, index):
        # Returns (xb, yb) pair
        idxs = torch.randint(0, 7488, (256,))
        X  = self.eeg[index][:,idxs]
        y =  self.y[index]
       
        return X, y
        
    def __len__(self):
        # Returns length
        return len(self.eeg)

class eeg_GAT(pl.LightningModule):

    def __init__(self,  A, nClasses=1, att=True):
        super(eeg_GAT, self).__init__()
        
        
        assert A is not None
        nNodes = A.shape[0]

        # Now, to create the proper graph object, since we're going to use
        # 'fuseEdges' option in createGraph, we are going to add an extra dimension
        # to the adjacencyMatrix (to indicate there's only one matrix in the 
        # collection that we should be fusing)
        #A = A.reshape([1, nNodes, nNodes])
            
        graphOptions = {} # Dictionary of options to pass to the createGraph function
        graphOptions['adjacencyMatrix'] = A
        G = graphTools.Graph('adjacency', nNodes, graphOptions)
        G.computeGFT() # Compute the eigendecomposition of the stored GSO
        
        
        S = G.S.copy()/np.max(np.real(G.E))
        
        modelGLGNN = {}
        
        device = 'cuda:0' if  torch.cuda.is_available() \
                                         else 'cpu'
                                         
        #\\\ ARCHITECTURE
            
        # Select architectural nn.Module to use
        #modelGLGNN['archit'] = archit.adaptiveGraphLearnGNN
        baseModel =  archit.GraphAttentionNetwork if att else archit.SelectionGNN 
        # Graph convolutional layers
        modelGLGNN['dimNodeSignals'] = [16, 8, 8] # Number of features per layer
        if att:
            modelGLGNN['nAttentionHeads'] = [3,3]
        else:
            modelGLGNN['nFilterTaps'] = [5, 5]
        modelGLGNN['bias'] = True # Include bias
        # Nonlinearity
        modelGLGNN['nonlinearity'] = nn.ReLU if not att else torch.nn.functional.relu
        # Pooling
        modelGLGNN['nSelectedNodes'] = [32, 32] # Number of nodes to keep se il secondo diventa 8 scoppia
        modelGLGNN['poolingFunction'] = gml.TransformerPool # Summarizing function
        modelGLGNN['poolingSize'] = [6, 8] # Summarizing neighborhoods
        # Readout layer
        modelGLGNN['dimLayersMLP'] = [nClasses]
        # Graph Structure
        modelGLGNN['GSO'] = S # To be determined later on, based on data
        modelGLGNN['order'] = None # To be determined next
        # Coarsening
        self.transformer = nn.Transformer(d_model=256, 
                                           nhead=1, 
                                           num_encoder_layers=1, 
                                           num_decoder_layers=1, 
                                           dim_feedforward=16, 
                                           batch_first=True).to("cpu")

        
        
   
        #modelGLGNN['coarsening'] = False
        
        #\\\ TRAINER

        
        #\\\ HPARAMS
        multipliers = { "lambda" : [0.005, 0.005], #Frob
                        "gamma" :  [0.03, 0.03], #Log-B
                        "beta" :   [5e-5, 5e-5], #TV
                        "eta" :    [5e-4, 6e-4]} # Enc-Dec
        self.attention = att
        
        self.arch = archit.adaptiveGraphLearnGNN(baseModel)(learn_first=True, **modelGLGNN).to("cpu")
        self.loss = MultiGraphLearningLoss(nn.L1Loss, self.arch.S, 
                                        self.arch.signals,
                                        self.arch.enc_dec_errors,
                                        multipliers)
        
    def forward(self, x):
        activation = self.transformer.encoder.layers[0].linear1(x)
        code = torch.relu(activation)
        code = code.permute(0,2,1)
        T = self.arch(code)
        out = T*((T>=0) & (T<=100))
        
        return  out
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        if torch.rand(1) > .5:
            self.loss.eval_penalties()
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        loss = self.loss(y_hat, y)
        
        
        
        self.log('val_loss', loss.item(), on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
      return self.validation_step(batch, batch_idx)
  

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-3, weight_decay=1e-6)
        return optimizer
    
    
net = eeg_GAT(np.ones((32,32)))
    


train_data = Dataset( )
val_data = Dataset( )

train_loader = \
    torch.utils.data.DataLoader(train_data, batch_size=10, batch_sampler=None, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, batch_sampler=None, shuffle=False)

logger = pl.loggers.TensorBoardLogger(name=r'1', save_dir='lightning_logs')


pl.seed_everything(0)



trainer = pl.Trainer(max_epochs=10, logger=logger)
trainer.fit(net, train_loader, val_loader)


s0 = net.arch.S[0]
s1 = net.arch.S[1]
s0 = s0[0]
s1 = s1[0]
import networkx as nx
G = nx.from_numpy_array(s0.numpy())
nx.draw(G)
G = nx.from_numpy_array(s1.numpy())
nx.draw(G)
