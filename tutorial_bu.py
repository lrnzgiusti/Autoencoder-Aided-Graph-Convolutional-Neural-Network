#%% Libs

import os
import numpy as np
import pickle
import datetime
from copy import deepcopy
import matplotlib
matplotlib.rcParams['text.usetex'] = False # Comment this line if no LaTeX installation is available
#matplotlib.rcParams['font.family'] = 'serif' # Comment this line if no LaTeX installation is available
#matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']
import matplotlib.pyplot as plt; plt.style.use('ggplot')
import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim
import alegnn.utils.graphTools as graphTools
import alegnn.utils.dataTools
import alegnn.utils.graphML as gml
import alegnn.modules.architectures as archit
import alegnn.modules.model as model
import alegnn.modules.training as training
import alegnn.modules.evaluation as evaluation
import alegnn.modules.loss as loss
from alegnn.utils.miscTools import writeVarValues
from alegnn.utils.miscTools import saveSeed



#%% Simulation Parameters 

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
    
    
    
useGPU = False

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



#%% Data parameters 
nTrain = 5000 # Number of training samples
nValid = int(0.2 * nTrain) # Number of validation samples
nTest = 50 # Number of testing samples
nNodes = 20 # Number of nodes
nClasses = 2 # Number of classes (i.e. number of communities)
graphOptions = {} # Dictionary of options to pass to the graphTools.createGraph function
graphOptions['nCommunities'] = nClasses # Number of communities
graphOptions['probIntra'] = 0.8 # Probability of drawing edges intra communities
graphOptions['probInter'] = 0.2 # Probability of drawing edges inter communities

tMax = None # Maximum number of diffusion times (W^t for t < tMax)
#\\\ Save values:
writeVarValues(varsFile, {'nNodes': nNodes, 'graphType': graphType})
writeVarValues(varsFile, graphOptions)
writeVarValues(varsFile, {'nTrain': nTest,
                          'nValid': nValid,
                          'nTest': nTest,
                          'tMax': tMax,
                          'nClasses': nClasses,
                          'useGPU': useGPU})



#%% Training parameters
lossFunction = nn.CrossEntropyLoss
trainer = training.Trainer
evaluator = evaluation.evaluate
optimAlg = 'ADAM'
learningRate = 0.001
beta1 = 0.9
beta2 = 0.999
nEpochs = 40 # Number of epochs
batchSize = 20 # Batch size
validationInterval = 20 # How many training steps to do the validationInterval

writeVarValues(varsFile,
               {'optimAlg': optimAlg,
                'learningRate': learningRate,
                'beta1': beta1,
                'lossFunction': lossFunction,
                'nEpochs': nEpochs,
                'batchSize': batchSize,
                'validationInterval': validationInterval})


#%% Architecture hyperparameters 
    
modelList = []
    

#%% Aggregation GNN 
hParamsAggGNN = {}
hParamsAggGNN['name'] = 'AggGNN' # We give a name to this architecture

hParamsAggGNN['nNodes'] = 1 # The nodes are selected starting from the 
    # top of the signal vector, for the order given in the data. Later
    # we reorder the data to follow the highest-degree criteria.
hParamsAggGNN['Nmax'] = None # If 'None' sets maxN equal to the size
    # of the graph, so that no information is lost when creating the
    # aggregation sequence z_{i}
    
hParamsAggGNN['order'] = 'Degree'
hParamsAggGNN['F'] = [1, 5, 5] # Features per layer (the first element is the number of input features)
hParamsAggGNN['K'] = [3, 3] # Number of filter taps per layer
hParamsAggGNN['bias'] = True # Decide whether to include a bias term
hParamsAggGNN['sigma'] = nn.ReLU # Selected nonlinearity
hParamsAggGNN['rho'] = nn.MaxPool1d # Pooling function
hParamsAggGNN['alpha'] = [2, 3] # Size of pooling function
hParamsAggGNN['dimLayersMLP'] = [nClasses]
writeVarValues(varsFile, hParamsAggGNN)
modelList += [hParamsAggGNN['name']]



#%% Selection GNN (with zero-padding) 
hParamsSelGNN = {} # Create the dictionary to save the hyperparameters
hParamsSelGNN['name'] = 'SelGNN' # Name the architecture

hParamsSelGNN['F'] = [1, 5, 5] # Features per layer (first element is the number of input features)
hParamsSelGNN['K'] = [3, 3] # Number of filter taps per layer
hParamsSelGNN['bias'] = True # Decide whether to include a bias term
hParamsSelGNN['sigma'] = nn.ReLU # Selected nonlinearity
hParamsSelGNN['rho'] = gml.MaxPoolLocal # Summarizing function
hParamsSelGNN['alpha'] = [2, 3] # alpha-hop neighborhood that
hParamsSelGNN['N'] = [10, 5] # Number of nodes to keep at the end of each layer is affected by the summary
hParamsSelGNN['order'] = 'Degree'
hParamsSelGNN['dimLayersMLP'] = [nClasses] # Dimension of the fully connected layers after the GCN layers
writeVarValues(varsFile, hParamsSelGNN)
modelList += [hParamsSelGNN['name']]
    
    
    

#%% Selection GNN (with graph-coarsening) 
hParamsCrsGNN = deepcopy(hParamsSelGNN)
hParamsCrsGNN['name'] = 'CrsGNN'
hParamsCrsGNN['rho'] = nn.MaxPool1d
hParamsCrsGNN['order'] = None # We don't need any special ordering, since
    # it will be determined by the hierarchical clustering algorithm

writeVarValues(varsFile, hParamsCrsGNN)
modelList += [hParamsCrsGNN['name']]
    

#%% Logging parameters 
# Parameters:
printInterval = 0 # After how many training steps, print the partial results
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


#%% Basic Setup


if useGPU and torch.cuda.is_available():
    device = 'cuda:0'
    torch.cuda.empty_cache()
else:
    device = 'cpu'
# Notify:
print("Device selected: %s" % device)

trainingOptions = {}

trainingOptions['saveDir'] = saveDir
trainingOptions['printInterval'] = printInterval
trainingOptions['validationInterval'] = validationInterval


#%% Graph Creation


G = graphTools.Graph(graphType, nNodes, graphOptions)
G.computeGFT()
sourceNodes = graphTools.computeSourceNodes(G.A, nClasses)
writeVarValues(varsFile, {'sourceNodes': sourceNodes})


#%% Data Creation

data = alegnn.utils.dataTools.SourceLocalization(G, nTrain, nValid, nTest, sourceNodes, tMax = tMax)
data.astype(torch.float64)
data.expandDims()


#%% Model Initialization
modelsGNN = {}


#%% Aggregation GNN


thisName = hParamsAggGNN['name']

#\\\ Architecture
thisArchit = archit.AggregationGNN(# Linear
                                   hParamsAggGNN['F'],
                                   hParamsAggGNN['K'],
                                   hParamsAggGNN['bias'],
                                   # Nonlinearity
                                   hParamsAggGNN['sigma'],
                                   # Pooling
                                   hParamsAggGNN['rho'],
                                   hParamsAggGNN['alpha'],
                                   # MLP in the end
                                   hParamsAggGNN['dimLayersMLP'],
                                   # Structure
                                   G.S/np.max(np.diag(G.E)), # Normalize the adjacency matrix
                                   order = hParamsAggGNN['order'],
                                   maxN = hParamsAggGNN['Nmax'],
                                   nNodes = hParamsAggGNN['nNodes'])

#\\\ Optimizer
thisOptim = optim.Adam(thisArchit.parameters(), lr = learningRate, betas = (beta1,beta2))

#\\\ Model
AggGNN = model.Model(thisArchit,
                     lossFunction(),
                     thisOptim,
                     trainer,
                     evaluator,
                     device,
                     thisName,
                     saveDir)

#\\\ Add model to the dictionary
modelsGNN[thisName] = AggGNN



#%% Selection GNN (with zero-padding) 
thisName = hParamsSelGNN['name']

#\\\ Architecture
thisArchit = archit.SelectionGNN(# Graph filtering
                                 hParamsSelGNN['F'],
                                 hParamsSelGNN['K'],
                                 hParamsSelGNN['bias'],
                                 # Nonlinearity
                                 hParamsSelGNN['sigma'],
                                 # Pooling
                                 hParamsSelGNN['N'],
                                 hParamsSelGNN['rho'],
                                 hParamsSelGNN['alpha'],
                                 # MLP
                                 hParamsSelGNN['dimLayersMLP'],
                                 # Structure
                                 G.S/np.max(np.real(G.E)), # Normalize adjacency
                                 order = hParamsSelGNN['order'])
# This is necessary to move all the learnable parameters to be
# stored in the device (mostly, if it's a GPU)
thisArchit.to(device)

#\\\ Optimizer
thisOptim = optim.Adam(thisArchit.parameters(), lr = learningRate, betas = (beta1,beta2))

#\\\ Model
SelGNN = model.Model(thisArchit,
                     lossFunction(),
                     thisOptim,
                     trainer,
                     evaluator,
                     device,
                     thisName,
                     saveDir)

#\\\ Add model to the dictionary
modelsGNN[thisName] = SelGNN



#%% Selection GNN (with graph coarsening)
thisName = hParamsCrsGNN['name']

#\\\ Architecture
thisArchit = archit.SelectionGNN(# Graph filtering
                                 hParamsCrsGNN['F'],
                                 hParamsCrsGNN['K'],
                                 hParamsCrsGNN['bias'],
                                 # Nonlinearity
                                 hParamsCrsGNN['sigma'],
                                 # Pooling
                                 hParamsCrsGNN['N'],
                                 hParamsCrsGNN['rho'],
                                 hParamsCrsGNN['alpha'],
                                 # MLP
                                 hParamsCrsGNN['dimLayersMLP'],
                                 # Structure
                                 G.S/np.max(np.real(G.E)),
                                 coarsening = True)
# This is necessary to move all the learnable parameters to be
# stored in the device (mostly, if it's a GPU)
thisArchit.to(device)

#\\\ Optimizer
thisOptim = optim.Adam(thisArchit.parameters(), lr = learningRate, betas = (beta1,beta2))

#\\\ Model
CrsGNN = model.Model(thisArchit,
                     lossFunction(),
                     thisOptim,
                     trainer,
                     evaluator,
                     device,
                     thisName,
                     saveDir)

#\\\ Add model to the dictionary
modelsGNN[thisName] = CrsGNN



#%% Training

lossTrain = {}
costTrain = {}
lossValid = {}
costValid = {}

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
    
#%% Evaluation


costBest = {} # Classification accuracy obtained for the best model
costLast = {} # Classification accuracy obtained for the last model

for thisModel in modelsGNN.keys():
    thisEvalVars = modelsGNN[thisModel].evaluate(data)
    
    costBest[thisModel] = thisEvalVars['costBest']
    costLast[thisModel] = thisEvalVars['costLast']
    
    
#%% Results


print("\nFinal evaluations")
for thisModel in modelList:
    print("\t%s: %6.2f%% [Best] %6.2f%% [Last]" % (
            thisModel,
            costBest[thisModel] * 100,
            costLast[thisModel] * 100))
    
#%% Figures 

#\\\ FIGURES DIRECTORY:
saveDirFigs = os.path.join(saveDir,'figs')
# If it doesn't exist, create it.
if not os.path.exists(saveDirFigs):
    os.makedirs(saveDirFigs)


# Get the value 'nBatches'
nBatches = thisTrainVars['nBatches']

# Compute the x-axis
xTrain = np.arange(0, nEpochs * nBatches, xAxisMultiplierTrain)
xValid = np.arange(0, nEpochs * nBatches, \
                      validationInterval*xAxisMultiplierValid)


        
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
    

    
    
    
    