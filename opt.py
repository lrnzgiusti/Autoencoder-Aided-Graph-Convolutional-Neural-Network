#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import sys

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1] 

import numpy as np
import pickle

import datetime
from copy import deepcopy

# %%
import matplotlib
matplotlib.rcParams['text.usetex'] = False # Comment this line if no LaTeX installation is available
#matplotlib.rcParams['font.family'] = 'serif' # Comment this line if no LaTeX installation is available
#matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# %%
import torch; torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.optim as optim

import random
#np.random.seed(10220033)
#random.seed(10220033)
#torch.manual_seed(10220033)


# %%
import alegnn.utils.graphTools as graphTools
import alegnn.utils.dataTools
import alegnn.utils.graphML as gml

import alegnn.modules.architectures as archit
import alegnn.modules.model as model
import alegnn.modules.training as training
import alegnn.modules.evaluation as evaluation
import alegnn.modules.loss as loss
import alegnn.modules.optim as GLOptim


# In[6]:


from alegnn.utils.miscTools import writeVarValues
from alegnn.utils.miscTools import saveSeed


graphType = 'SBM' # Type of graph
thisFilename = 'sourceLocTutorial' # This is the general name of all related files
saveDirRoot = 'experiments' # Relative location where to save the file
saveDir = os.path.join(saveDirRoot, thisFilename) # Dir where to save all the results from each run

#\\\ Create .txt to store the values of the setting parameters for easier
# reference when running multiple experiments
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# Append date and time of the run to the directory, to avoid several runs of
# overwritting each other.
saveDir = saveDir + '-' + graphType + '-' + today
# Create directory
if not os.path.exists(saveDir):
    os.makedirs(saveDir)


# Create the file where all the (hyper)parameters are results will be saved.
varsFile = os.path.join(saveDir,'hyperparameters.txt')
with open(varsFile, 'w+') as file:
    file.write('%s\n\n' % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))


# Now, decide if we are going to use the GPU or not.

# %%


useGPU = True



#\\\ Save seeds for reproducibility
#   PyTorch seeds
torchState = torch.get_rng_state()
torchSeed = torch.initial_seed()
#   Numpy seeds
numpyState = np.random.RandomState().get_state()
#   Collect all random states
randomStates = []
randomStates.append({})
randomStates[0]['module'] = 'numpy'
randomStates[0]['state'] = numpyState
randomStates.append({})
randomStates[1]['module'] = 'torch'
randomStates[1]['state'] = torchState
randomStates[1]['seed'] = torchSeed
#   This list and dictionary follows the format to then be loaded, if needed,
#   by calling the loadSeed function in Utils.miscTools
saveSeed(randomStates, saveDir)



# %%


nTrain = 5000 # Number of training samples
nValid = int(0.2 * nTrain) # Number of validation samples
nTest = 50 # Number of testing samples


# Then, the number of nodes and the number of communities of the stochastic block model graph (that was selected before when defining <code>graphType = 'SBM'</code>). Recall that the objective of the problem is to determine which community originated the diffusion, and as such, the number of communities is equal to the number of classes.

# %%


nNodes = 20 # Number of nodes
nClasses = 2 # Number of classes (i.e. number of communities)
graphOptions = {} # Dictionary of options to pass to the graphTools.createGraph function
graphOptions['nCommunities'] = nClasses # Number of communities
graphOptions['probIntra'] = 0.8 # Probability of drawing edges intra communities
graphOptions['probInter'] = 0.2 # Probability of drawing edges inter communities


# %%


tMax = None # Maximum number of diffusion times (W^t for t < tMax)



# %%


#\\\ Save values:
writeVarValues(varsFile, {'nNodes': nNodes, 'graphType': graphType})
writeVarValues(varsFile, graphOptions)
writeVarValues(varsFile, {'nTrain': nTest,
                          'nValid': nValid,
                          'nTest': nTest,
                          'tMax': tMax,
                          'nClasses': nClasses,
                          'useGPU': useGPU})


# %%


lossFunction = nn.CrossEntropyLoss
customLoss = loss.MultiGraphLearningLoss



# Now that we have selected the loss function, we need to determine how to handle the training and evaluation. This, mostly, amounts to selecting wrappers that will handle the batch size partitioning, early stopping, validation, etc. The specifics of the evaluation measure, for example, depend on the data being measured and, thus, are parte of the <code>data</code> class.

# %%


trainer = training.Trainer
evaluator = evaluation.evaluate
multiTaskTrainer = training.MultiTaskTrainer

# Next, we determine the optimizer we use with all its parameters. In our case, an ADAM optimizer, where the variables <code>beta1</code> and <code>beta2</code> are the forgetting factors $\beta_{1}$ and $\beta_{2}$.

# %%


optimAlg = 'ADAM'
learningRate = 0.001
beta1 = 0.9
beta2 = 0.999


# Finally, we determine the training process. The number of epochs, the batch size, and how often we carry out validation (i.e. after how many update steps, we run a validation step).

# In[18]:

B = np.random.choice([16,32,64,128,256])

nEpochs = 50 # Number of epochs
batchSize = B # Batch size
validationInterval = 20 # How many training steps to do the validation


# Save the values into the <code>.txt</code> file.

# In[19]:


writeVarValues(varsFile,
                {'optimAlg': optimAlg,
                'learningRate': learningRate,
                'beta1': beta1,
                'lossFunction': lossFunction,
                'nEpochs': nEpochs,
                'batchSize': batchSize,
                'validationInterval': validationInterval})


# In[20]:


F1 = 2**np.random.randint(1, 6) #used for filter size and filter taps
F2 = 2**np.random.randint(1, 6)

K1 = np.random.randint(1, 5)
K2 = np.random.randint(1, 5)

A1 = np.random.randint(1, 5)
A2 = np.random.randint(1, 5)

N1 = np.random.randint(1, 20)
N2 = np.random.randint(1, N1)



modelList = []



# In[32a]:
hParamsGLGNN = {} # Create the dictionary to save the hyperparameters
hParamsGLGNN['name'] = 'GLGNN' # Name the architecture


hParamsGLGNN['F'] = [1, F1, F2] # Features per layer (first element is the number of input features)
hParamsGLGNN['K'] = [K1, K2] # Number of filter taps per layer
hParamsGLGNN['bias'] = True # Decide whether to include a bias term
hParamsGLGNN['sigma'] =  nn.LeakyReLU # nn.ReLU # Selected nonlinearity  #              
hParamsGLGNN['rho'] = gml.EncDecPool # Summarizing function
hParamsGLGNN['alpha'] = [A1, A2] # alpha-hop neighborhood that
hParamsGLGNN['N'] = [N1, N2] # Number of nodes to keep at the end of each layer is affected by the summary


hParamsGLGNN['order'] = 'SpectralProxies' #'Degree' OBSOLETE
hParamsGLGNN['dimLayersMLP'] = [nClasses] # Dimension of the fully connected layers after the GCN layers


writeVarValues(varsFile, hParamsGLGNN)
modelList += [hParamsGLGNN['name']]


# %%


# Parameters:
printInterval = 100 # After how many training steps, print the partial results
    # if 0 never print training partial results.
xAxisMultiplierTrain = 100 # How many training steps in between those shown in
    # the plot, i.e., one training step every xAxisMultiplierTrain is shown.
xAxisMultiplierValid = 10 # How many validation steps in between those shown,
    # same as above.
figSize = 5 # Overall size of the figure that contains the plot
lineWidth = 2 # Width of the plot lines
markerShape = 'o' # Shape of the markers
markerSize = 3 # Size of the markers

writeVarValues(varsFile,
               {'saveDir': saveDir,
                'printInterval': printInterval,
                'figSize': figSize,
                'lineWidth': lineWidth,
                'markerShape': markerShape,
                'markerSize': markerSize})


# ## Basic Setup <a class="anchor" id="sec:basicSetup"></a>
# Set the basic setup with the parameters chosen above, preparing the field for the upcoming data generation and training procedures.

# Determine the processing unit.

# In[35]:


if useGPU and torch.cuda.is_available():
    device = 'cuda:0'
    torch.cuda.empty_cache()
else:
    device = 'cpu'
# Notify:
print("Device selected: %s" % device)


# Save the training options in a dictionary that is then passed onto the training function.

# In[36]:


trainingOptions = {}

trainingOptions['saveDir'] = saveDir
trainingOptions['printInterval'] = printInterval
trainingOptions['validationInterval'] = validationInterval


# ## Graph Creation <a class="anchor" id="sec:graphCreation"></a>
# Let's create the graph that we use as support for the source localization problem. The graph will be created as a <code>Graph</code> class that is available in <code>Utils.graphTools</code>. This <code>Graph</code> class has several useful attributes and methods, and can create different types of graph. In particular, we want to create an SBM graph with <code>nNodes</code> nodes, and with the <code>graphOptions</code> specified in the above determined parameters (number of communities, probabilities of drawing edges within the same community and to external communities).

# In[37]:


G = graphTools.Graph(graphType, nNodes, graphOptions)

# With this initialization, the <code>Graph</code> class <code>G</code> contains several useful attributes (like the adjacency matrix, the diagonal matrix, flags to signal if the graph is undirected and has self-loops, and also the graph Laplacian -if the graph is undirected and has no self-loops-). More importantly, it has a GSO attributes <code>G.S</code> that stores the selected GSO (by default is the adjacency matrix, but can be changed by using the method <code>G.setGSO</code>; for instance, if we want to use the graph Laplacian instead, we call <code>G.setGSO(G.L)</code>).
# 
# Once we have created the graph (which is a realization of an SBM random graph), we can compute the GFT of the stored GSO.

# In[38]:


G.computeGFT()


# We note that, by default, the GFT is computed on the stored GSO (located at <code>G.S</code>) and orders the resulting eigenvalues following the total variation order. A different order can be computed when setting a new GSO; for instance, if we want to set the graph Laplacian as the GSO and order it in increasing eigenvalues, we call <code>G.setGSO(G.L, GFT = 'increasing')</code>.

# Once we have the graph, we need to compute which nodes are the source nodes for the diffusion. For this, we have the function <code>computeSourceNodes</code> in the <code>Utils.graphTools</code> library. This function takes the adjacency matrix of the graph and the number of classes, and computes a spectral clustering. Then, selects the node with the highest degree within each cluster and assigns that as the source node, representative of the cluster. We note that, in the case of an SBM graph, this clustering coincides with the communities, and that all nodes, within the same community, have the same expected degree.

# In[39]:


sourceNodes = graphTools.computeSourceNodes(G.A, nClasses)


# We save the list of selected source nodes.

# In[40]:


writeVarValues(varsFile, {'sourceNodes': sourceNodes})


# ## Data Creation <a class="anchor" id="sec:dataCreation"></a>
# Now that we have the graph and the nodes that we use as sources, we can proceed to create the [datasets](#subsec:sourceLoc). Each sample in the dataset is created following
# 
# $$ \mathbf{x} = \mathbf{W}^{t} \boldsymbol{\delta}_{c} $$
# 
# where $\mathbf{W}$ represents the adjacency matrix and $\boldsymbol{\delta}_{c}$ is a graph signal that has a $1$ in node $c$ and $0$ elsewhere. The source node $c$ is selected at random from the <code>sourceNodes</code> list obtained above, and the value of $t$ is selected at random between <code>0</code> and the value of <code>tMax</code> determined in the data parameters [above](#subsec:dataParameters). (We use the adjacency, normalized by the largest eigenvalue, for numerical stability).
# 
# The datasets are created into <code>data</code> which is an instance of the <code>SourceLocalization</code> class that is defined in the library <code>Utils.dataTools</code>. This has several useful methods for handling the data while training. The initialization of this class (the creation of the dataset) takes the following inputs: the graph <code>G</code>, the number of samples in each of the training, validation, and testing sets, <code>nTrain</code>, <code>nValid</code> and <code>nTest</code>, respectively (which were defined [above](#subsec:dataParameters) in the parameter definition section), the list of source nodes <code>sourceNodes</code> that we just computed, and the value of <code>tMax</code> (that was also defined in the parameter definition section). After creating the data, we transform it into <code>torch.float64</code> type, since we're going to use PyTorch for training the models. Note that the dataset created is of shape <code>nSamples</code> x <code>nNodes</code>, but that the architectures assume an input that is a graph signal, that is, that it has shape <code>nSamples</code> x <code>nFeatures</code> x <code>nNodes</code>. Therefore, we need to add an extra dimension for the features (since <code>nFeatures = 1</code>) and make the dataset of shape <code>nSamples</code> x <code>1</code> x <code>nNodes</code>. This is achieved by using the method <code>.expandDims()</code> that belongs to the <code>data</code> class.

# In[41]:


data = alegnn.utils.dataTools.SourceLocalization(G, nTrain, nValid, nTest, sourceNodes, tMax = tMax)
data.astype(torch.float64)
data.expandDims()

#data.samples['train']['signals'] += torch.randn_like(data.samples['train']['signals']) * 0.025
#data.samples['valid']['signals'] += torch.randn_like(data.samples['valid']['signals']) * 0.025
#data.samples['test']['signals'] += torch.randn_like(data.samples['test']['signals']) * 0.025




# %%


modelsGNN = {}    

bag_of_parameters=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7,
                    3e-1, 3e-2, 3e-3, 3e-4, 3e-5, 3e-6, 3e-7,
                    5e-1, 5e-2, 5e-3, 5e-4, 5e-5, 5e-6, 5e-7,
                    7e-1, 7e-2, 7e-3, 7e-4, 7e-5, 7e-6, 7e-7,
                    9e-1, 9e-2, 9e-3, 9e-4, 9e-5, 9e-6, 9e-7]

lr, lam, gam, bet, eta = np.random.choice(bag_of_parameters,
                                            5, replace=True)

multipliers={"lambda" : [0.0, lam], #Frob
            "gamma" : [0.0, gam], #Log-B
            "beta" : [0.0, bet],
            "eta" :[eta]} # TV





thisName = hParamsGLGNN['name']

#\\\ Architecture
thisArchit = archit.GraphLearnGNN(
                                 hParamsGLGNN['F'],
                                 hParamsGLGNN['K'],
                                 hParamsGLGNN['bias'],
                                 # Nonlinearity
                                 hParamsGLGNN['sigma'],
                                 # Pooling
                                 hParamsGLGNN['N'],
                                 hParamsGLGNN['rho'],
                                 hParamsGLGNN['alpha'],
                                 # MLP
                                 hParamsGLGNN['dimLayersMLP'],
                                 # Structure
                                 G.S/np.max(np.real(G.E)), # Normalize adjacency
                                 order = hParamsGLGNN['order'])
# This is necessary to move all the learnable parameters to be
# stored in the device (mostly, if it's a GPU)
thisArchit.to(device)

#\\\ Optimizer
thisOptim = GLOptim.MultiGraphLearningOptimizer(thisArchit.named_parameters(), 
                                                thisArchit.constants,
                                                lr = lr,
                                                betas = (beta1,beta2),
                                                momentum = 0.0)
 
#\\\ Model
GLGNN = model.Model(thisArchit,
                     customLoss(nn.CrossEntropyLoss, 
                                thisArchit.S, 
                                thisArchit.signals,
                                thisArchit.enc_dec_errors,
                                multipliers),
                     thisOptim,
                     trainer, #multiTaskTrainer,
                     evaluator,
                     device,
                     thisName,
                     saveDir)

#\\\ Add model to the dictionary
modelsGNN[thisName] = GLGNN
with open(varsFile, 'a+') as file:
    file.write(repr(GLGNN))
# %%


lossTrain = {}
costTrain = {}
lossValid = {}
costValid = {}


# And now we can indeed train the models

# In[47]:

#modelList = ['AggGNN', 'MultiGNN', 'GLGNN']
#del modelsGNN['SelGNN']
#del modelsGNN['CrsGNN']
for thisModel in modelsGNN.keys():
    print("Training model %s..." % thisModel, end = ' ', flush = True)
    
    #Train
    thisTrainVars = modelsGNN[thisModel].train(data, nEpochs, batchSize, **trainingOptions)
    # Save the variables
    lossTrain[thisModel] = thisTrainVars['lossTrain']
    costTrain[thisModel] = thisTrainVars['costTrain']
    lossValid[thisModel] = thisTrainVars['lossValid']
    costValid[thisModel] = thisTrainVars['costValid']
    
    print("OK", flush = True)

# %% 

costBest = {} # Classification accuracy obtained for the best model
costLast = {} # Classification accuracy obtained for the last model



# Then, we just run the <code>.evaluate</code> method from the <code>Model</code> class. The output is a dictionary with the corresponding cost, which we save into the previous dictionary.

# In[49]:


for thisModel in modelsGNN.keys():
    thisEvalVars = modelsGNN[thisModel].evaluate(data)
    
    costBest[thisModel] = thisEvalVars['costBest']
    costLast[thisModel] = thisEvalVars['costLast']


# Now, test each of the models in the dictionary on this data, and save the results.

# ## Results <a class="anchor" id="sec:results"></a>
# Now, we process the results. We print them, the evaluation results on the test set, and we also create and save figures for the training stage, depicting both the loss and the evaluation values throughout training.

# In[50]:


print("\nFinal evaluations")
for thisModel in modelList:
    print("\t%s: %6.2f%% [Best] %6.2f%% [Last]" % (
            thisModel,
            costBest[thisModel] * 100,
            costLast[thisModel] * 100))


# ### Figures <a class="anchor" id="subsec:figures"></a>
# The figures that are going to be plotted are: The loss function during training (for both the training set and the validation set) for each of the models, the evaluation function during training (for both the validation set and the training set) for each of the models, the loss function during training on the training set for all models at once, and the evaluation function during training on the validation set for all models at once.

# First, we create the directory where to save the plots.

# In[51]:


#\\\ FIGURES DIRECTORY:
saveDirFigs = os.path.join(saveDir,'figs')
# If it doesn't exist, create it.
if not os.path.exists(saveDirFigs):
    os.makedirs(saveDirFigs)


# And we create the plots. Starting with computing the x-axis (if there are many training points, the x-axis might become too crowded, and therefore we just want to downsample the plot).

# In[52]:


# Get the value 'nBatches'
nBatches = thisTrainVars['nBatches']

# Compute the x-axis
xTrain = np.arange(0, nEpochs * nBatches, xAxisMultiplierTrain)
xValid = np.arange(0, nEpochs * nBatches,                       validationInterval*xAxisMultiplierValid)

# If we do not want to plot all the elements (to avoid overcrowded plots)
# we need to recompute the x axis and take those elements corresponding
# to the training steps we want to plot
if xAxisMultiplierTrain > 1:
    # Actual selected samples
    selectSamplesTrain = xTrain
    # Go and fetch tem
    for thisModel in modelList:
        lossTrain[thisModel] = np.log(lossTrain[thisModel][selectSamplesTrain]+1e-8)
        costTrain[thisModel] = np.log(costTrain[thisModel][selectSamplesTrain]+1e-8)
# And same for the validation, if necessary.
if xAxisMultiplierValid > 1:
    selectSamplesValid = np.arange(0, len(lossValid[thisModel]),                                    xAxisMultiplierValid)
    for thisModel in modelList:
        lossValid[thisModel] = np.log(lossValid[thisModel][selectSamplesValid]+1e-8)
        costValid[thisModel] = np.log(costValid[thisModel][selectSamplesValid]+1e-8)


# Plot the training and validation loss (one figure for each model)

# In[76]:
matplotlib.rcParams['text.usetex'] = False

for key in lossTrain.keys():
    lossFig = plt.figure(figsize=(1.61*figSize, 1*figSize))
    plt.plot(xTrain, lossTrain[key],
             color = '#01256E', linewidth = lineWidth,
             marker = markerShape, markersize = markerSize)
    plt.plot(xValid, lossValid[key],
             color = '#95001A', linewidth = lineWidth,
             marker = markerShape, markersize = markerSize)
    plt.ylabel(r'Loss')
    plt.xlabel(r'Training steps')
    plt.legend([r'Training', r'Validation'])
    plt.title(r'%s' % key)
    lossFig.savefig(os.path.join(saveDirFigs,'loss%s.pdf' % key),
                    bbox_inches = 'tight')


# Plot the evaluation measure (the classification accuracy) on both the training and validation set (one figure for each model)

# In[77]:


for key in costTrain.keys():
    costFig = plt.figure(figsize=(1.61*figSize, 1*figSize))
    plt.plot(xTrain, costTrain[key],
             color = '#01256E', linewidth = lineWidth,
             marker = markerShape, markersize = markerSize)
    plt.plot(xValid, costValid[key],
             color = '#95001A', linewidth = lineWidth,
             marker = markerShape, markersize = markerSize)
    plt.ylabel(r'Error rate')
    plt.xlabel(r'Training steps')
    plt.legend([r'Training', r'Validation'])
    plt.title(r'%s' % key)
    costFig.savefig(os.path.join(saveDirFigs,'eval%s.pdf' % key),
                    bbox_inches = 'tight')


# Plot the loss on the training set for all models (one figure, for comparison between models).

# In[78]:


allLossTrain = plt.figure(figsize=(1.61*figSize, 1*figSize))
for key in lossTrain.keys():
    plt.plot(xTrain, lossTrain[key],
             linewidth = lineWidth,
             marker = markerShape, markersize = markerSize)
plt.ylabel(r'Loss')
plt.xlabel(r'Training steps')
plt.legend(list(lossTrain.keys()))
allLossTrain.savefig(os.path.join(saveDirFigs,'allLossTrain.pdf'),
                bbox_inches = 'tight')


# Plot the evaluation measure (classification accuracy) on the validation set for all models (one figure).

# In[79]:


allEvalValid = plt.figure(figsize=(1.61*figSize, 1*figSize))
for key in costValid.keys():
    plt.plot(xValid, costValid[key],
             linewidth = lineWidth,
             marker = markerShape, markersize = markerSize)
plt.ylabel(r'Error rate')
plt.xlabel(r'Training steps')
plt.legend(list(costValid.keys()))
allEvalValid.savefig(os.path.join(saveDirFigs,'allEvalValid.pdf'),
                bbox_inches = 'tight')


# ## Conclusion <a class="anchor" id="sec:conclusions"></a>
# This concludes the tutorial. The main objective was to introduce the basic call to the architectures. Additionally, we reviewed the classes for graphs and datasets that could be useful. In particular, the data class has important attributes and methods that can work in tandem with training multiple models.
# 
# While only three architectures were overviewed in this tutorial, many more are available in <code>Modules.architectures</code>. Additionally, the <code>nn.Module</code>s defined in <code>Utils.graphML</code> can serve as basic layers for building other graph neural network architectures tailored to specific needs, hopefully in the same way as the layers in <code>torch.nn</code> are typically used.
# In[80]:
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

    
json.dump(costValid, open('Err_'+str(np.random.randint(0,1000))+'.json', 'w'), indent=4, cls=NumpyEncoder)