# 2018/12/03~
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu

# Simulate the source localization problem. We have a graph, and we observe a
# signal defined on top of this graph. This signal is assumed to represent the
# diffusion of a rumor. The rumor is observed after being diffused for an
# unknown amount of time. The objective is to determine which is the node (or 
# the community) that started the rumor.

# Outputs:
# - Text file with all the hyperparameters selected for the run and the 
#   corresponding results (hyperparameters.txt)
# - Pickle file with the random seeds of both torch and numpy for accurate
#   reproduction of results (randomSeedUsed.pkl)
# - The parameters of the trained models, for both the Best and the Last
#   instance of each model (savedModels/)
# - The figures of loss and evaluation through the training iterations for
#   each model (figs/ and trainVars/)
# - If selected, logs in tensorboardX certain useful training variables

#%%##################################################################
#                                                                   #
#                    IMPORTING                                      #
#                                                                   #
#####################################################################

#\\\ Standard libraries:
import os
import sys

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1] 

import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = False
#matplotlib.rcParams['font.family'] = 'serif'
#matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pickle
import datetime
from copy import deepcopy

import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim

#\\\ Own libraries:
import alegnn.utils.graphTools as graphTools
import alegnn.utils.dataTools
import alegnn.utils.graphML as gml
import alegnn.modules.architectures as archit
import alegnn.modules.model as model
import alegnn.modules.training as training
import alegnn.modules.evaluation as evaluation
import alegnn.modules.loss as loss
import alegnn.modules.optim as GLOptim

#\\\ Separate functions:
from alegnn.utils.miscTools import writeVarValues
from alegnn.utils.miscTools import saveSeed

# Start measuring time
startRunTime = datetime.datetime.now()

#%%##################################################################
#                                                                   #
#                    SETTING PARAMETERS                             #
#                                                                   #
#####################################################################

graphType = "20news"  # Type of graph: 'SBM', 'FacebookEgo', 'SmallWorld'

thisFilename = 'Text-Classif' # This is the general name of all related files

saveDirRoot = 'experiments' # In this case, relative location
saveDir = os.path.join(saveDirRoot, thisFilename) # Dir where to save all
    # the results from each run
if graphType == 'FacebookEgo':
    dataDir = os.path.join('datasets','facebookEgo')

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

########
# DATA #
########

useGPU = True # If true, and GPU is available, use it.
############
# DATASETS #
############

data = alegnn.utils.dataTools.TwentyNews(0.08, 1000, 20, 16, 'cosine', None)
nTrain = data.samples['train']['signals'].shape[0] # Number of training samples
nValid = data.samples['valid']['signals'].shape[0] # Number of validation samples
nTest = data.samples['test']['signals'].shape[0] # Number of testing samples
tMax = 25 # Maximum number of diffusion times (A^t for t < tMax)

nDataRealizations = 1 # Number of data realizations
nGraphRealizations = 1 # Number of graph realizations
nClasses = data.samples['test']['targets'].max()+1 # Number of source nodes to select

nNodes = data.samples['train']['signals'].shape[-1] # Number of nodes

graphOptions = {} # Dictionary of options to pass to the createGraph function
graphOptions['isolatedNodes'] = False # If True keeps isolated nodes
graphOptions['forceConnected'] = True # If True removes nodes (from lowest to highest degree)
graphOptions['adjacencyMatrix'] = data.adjacencyMatrix

G = graphTools.Graph('adjacency', nNodes, graphOptions)
G.computeGFT() # Compute the eigendecomposition of the stored GSO

nNodes = G.N
data.astype(torch.float64)
data.expandDims()

data.to('cuda:0' if (useGPU and torch.cuda.is_available()) \
                                    else 'cpu')


if graphType == 'SBM':
    graphOptions['nCommunities'] = nClasses # Number of communities
    graphOptions['probIntra'] = 0.8 # Intracommunity probability
    graphOptions['probInter'] = 0.2 # Intercommunity probability
elif graphType == 'SmallWorld':
    graphOptions['probEdge'] = 0.5 # Edge probability
    graphOptions['probRewiring'] = 0.1 # Probability of rewiring
elif graphType == 'FacebookEgo':
    graphOptions['isolatedNodes'] = False # If True keeps isolated nodes
    graphOptions['forceConnected'] = True # If True removes nodes (from lowest to highest degree)
        # until the resulting graph is connected.
    use234 = True # Use a smaller 234-matrix with 2-communities instead of the full
        # graph with around 4k users
    nGraphRealizations = 1 # Number of graph realizations
    if use234:
        nClasses = 2

#\\\ Save values:
writeVarValues(varsFile, {'nNodes': nNodes, 'graphType': graphType})
writeVarValues(varsFile, graphOptions)
writeVarValues(varsFile, {'nTrain': nTrain,
                          'nValid': nValid,
                          'nTest': nTest,
                          'tMax': tMax,
                          'nDataRealizations':nDataRealizations,
                          'nGraphRealizations': nGraphRealizations,
                          'nClasses': nClasses,
                          'useGPU': useGPU})

############
# TRAINING #
############

#\\\ Individual model training options
optimAlg = 'ADAM' # Options: 'SGD', 'ADAM', 'RMSprop'
learningRate = 0.001 # In all options
beta1 = 0.9 # beta1 if 'ADAM', alpha if 'RMSprop'
beta2 = 0.999 # ADAM option only

#\\\ Loss function choice
lossFunction = nn.CrossEntropyLoss # This applies a softmax before feeding
    # it into the NLL, so we don't have to apply the softmax ourselves.

B = int(sys.argv[2])
lr = float(sys.argv[3])



#\\\ Overall training options
nEpochs = 300 # Number of epochs
batchSize = B # Batch size
doLearningRateDecay = True # Learning rate decay
learningRateDecayRate = 0.9 # Rate
learningRateDecayPeriod = 30 # How many epochs after which update the lr
validationInterval = 5 # How many training steps to do the validation

#\\\ Save values
writeVarValues(varsFile,
               {'optimAlg': optimAlg,
                'learningRate': learningRate,
                'beta1': beta1,
                'lossFunction': lossFunction,
                'nEpochs': nEpochs,
                'batchSize': batchSize,
                'doLearningRateDecay': doLearningRateDecay,
                'learningRateDecayRate': learningRateDecayRate,
                'learningRateDecayPeriod': learningRateDecayPeriod,
                'validationInterval': validationInterval})

#################
# ARCHITECTURES #
#################

# These will be two-layers Selection and Aggregation with pooling and different
# orderings.    

# Select pooling options (node ordering for zero-padding)
doDegree = True
doSpectralProxies = True
doEDS = True
doCoarsening = True

# Select desired architectures
doSelectionGNN = False
doAggregationGNN = False
doMultinodeGNN = False
doGraphLearnGNN = True

# In this section, we determine the (hyper)parameters of models that we are
# going to train. This only sets the parameters. The architectures need to be
# created later below. Do not forget to add the name of the architecture
# to modelList.

# If the model dictionary is called 'model' + name, then it can be
# picked up immediately later on, and there's no need to recode anything after
# the section 'Setup' (except for setting the number of nodes in the 'N' 
# variable after it has been coded).

# The name of the keys in the model dictionary have to be the same
# as the names of the variables in the architecture call, because they will
# be called by unpacking the dictionary.

modelList = []

device = 'cuda:0' if (useGPU and torch.cuda.is_available()) else 'cpu'

data.to(device)
#\\\\\\\\\\\\\\\\\\\\\
#\\ Graph Learn GNN \\
#\\\\\\\\\\\\\\\\\\\\\

if doGraphLearnGNN:
    modelGLGNN = {}
    modelGLGNN['name'] = 'GLGNN' # To be modified later on depending on the
        # specific ordering selected
    modelGLGNN['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) \
                                     else 'cpu'
                                     
    #\\\ ARCHITECTURE
        
    # Select architectural nn.Module to use
    modelGLGNN['archit'] = archit.GraphLearnGNN
    # Graph convolutional layers
    modelGLGNN['dimNodeSignals'] = [1, 32, 64] # Number of features per layer
    modelGLGNN['nFilterTaps'] = [5, 5] # Number of filter taps
    modelGLGNN['bias'] = True # Include bias
    # Nonlinearity
    modelGLGNN['nonlinearity'] = nn.ReLU
    # Pooling
    modelGLGNN['nSelectedNodes'] = [250, 100] # Number of nodes to keep
    modelGLGNN['poolingFunction'] = gml.EncDecPool # Summarizing function
    modelGLGNN['poolingSize'] = [6, 8] # Summarizing neighborhoods
    # Readout layer
    modelGLGNN['dimLayersMLP'] = [nClasses]
    # Graph Structure
    modelGLGNN['GSO'] = None # To be determined later on, based on data
    modelGLGNN['order'] = None # To be determined next
    # Coarsening
    #modelGLGNN['coarsening'] = False
    
    #\\\ TRAINER

    modelGLGNN['trainer'] = training.Trainer
    
    #\\\ HPARAMS
    multipliers = { "lambda" : [0.0, 5e-6], #Frob
                    "gamma" :  [0.0, 3e-7], #Log-B
                    "beta" :   [3e-6, 0.0], #TV
                    "eta" :    [1e-3, 1e-3]} # Enc-Dec
    print(multipliers)

    #\\\ EVALUATOR
    
    modelGLGNN['evaluator'] = evaluation.evaluate    
    writeVarValues(varsFile, modelGLGNN)
    modelList += [modelGLGNN['name']]
#\\\\\\\\\\\\\\\\\\\\\
#\\\ SELECTION GNN \\\
#\\\\\\\\\\\\\\\\\\\\\

# Hyperparameters to be shared by all Selection GNN architectures

if doSelectionGNN:
    
    #\\\ Basic parameters for all the Selection GNN architectures
    
    modelSelGNN = {}
    modelSelGNN['name'] = 'SelGNN' # To be modified later on depending on the
        # specific ordering selected
    modelSelGNN['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) \
                                     else 'cpu'
                                     
    #\\\ ARCHITECTURE
        
    # Select architectural nn.Module to use
    modelSelGNN['archit'] = archit.SelectionGNN
    # Graph convolutional layers
    modelSelGNN['dimNodeSignals'] = [1, 32, 32] # Number of features per layer
    modelSelGNN['nFilterTaps'] = [5, 5] # Number of filter taps
    modelSelGNN['bias'] = True # Include bias
    # Nonlinearity
    modelSelGNN['nonlinearity'] = nn.ReLU
    # Pooling
    modelSelGNN['nSelectedNodes'] = [10, 10] # Number of nodes to keep
    modelSelGNN['poolingFunction'] = gml.MaxPoolLocal # Summarizing function
    modelSelGNN['poolingSize'] = [6, 8] # Summarizing neighborhoods
    # Readout layer
    modelSelGNN['dimLayersMLP'] = [nClasses]
    # Graph Structure
    modelSelGNN['GSO'] = None # To be determined later on, based on data
    modelSelGNN['order'] = None # To be determined next
    # Coarsening
    modelSelGNN['coarsening'] = False
    
    #\\\ TRAINER

    modelSelGNN['trainer'] = training.Trainer
    
    #\\\ EVALUATOR
    
    modelSelGNN['evaluator'] = evaluation.evaluate
    

#\\\\\\\\\\\\
#\\\ MODEL 1: Selection GNN with nodes ordered by degree
#\\\\\\\\\\\\

if doSelectionGNN and doDegree:

    modelSelGNNdeg = deepcopy(modelSelGNN)

    modelSelGNNdeg['name'] += 'deg' # Name of the architecture
    # Structure
    modelSelGNNdeg['order'] = 'Degree'
    
    #\\\ Save Values:
    writeVarValues(varsFile, modelSelGNNdeg)
    modelList += [modelSelGNNdeg['name']]
    
#\\\\\\\\\\\\
#\\\ MODEL 2: Selection GNN with nodes ordered by EDS
#\\\\\\\\\\\\

if doSelectionGNN and doEDS:

    modelSelGNNeds = deepcopy(modelSelGNN)

    modelSelGNNeds['name'] += 'eds' # Name of the architecture
    # Structure
    modelSelGNNeds['order'] = 'EDS'
    
    #\\\ Save Values:
    writeVarValues(varsFile, modelSelGNNeds)
    modelList += [modelSelGNNeds['name']]
    
#\\\\\\\\\\\\
#\\\ MODEL 3: Selection GNN with nodes ordered by spectral proxies
#\\\\\\\\\\\\

if doSelectionGNN and doSpectralProxies:

    modelSelGNNspr = deepcopy(modelSelGNN)

    modelSelGNNspr['name'] += 'spr' # Name of the architecture
    # Structure
    modelSelGNNspr['order'] = 'SpectralProxies'
    
    #\\\ Save Values:
    writeVarValues(varsFile, modelSelGNNspr)
    modelList += [modelSelGNNspr['name']]
    
#\\\\\\\\\\\\
#\\\ MODEL 4: Selection GNN with graph coarsening
#\\\\\\\\\\\\

if doSelectionGNN and doCoarsening:

    modelSelGNNcrs = deepcopy(modelSelGNN)

    modelSelGNNcrs['name'] += 'crs' # Name of the architecture
    modelSelGNNcrs['poolingFunction'] = nn.MaxPool1d
    modelSelGNNcrs['poolingSize'] = [2, 2]
    modelSelGNNcrs['coarsening'] = True

    #\\\ Save Values:
    writeVarValues(varsFile, modelSelGNNcrs)
    modelList += [modelSelGNNcrs['name']]
    
#\\\\\\\\\\\\\\\\\\\\\\\
#\\\ AGGREGATION GNN \\\
#\\\\\\\\\\\\\\\\\\\\\\\

if doAggregationGNN:
    
    #\\\ Basic parameters for all the Aggregation GNN architectures
    
    modelAggGNN = {}
    
    modelAggGNN['name'] = 'AggGNN' # To be modified later on depending on the
        # specific ordering selected
    modelAggGNN['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) \
                                     else 'cpu'
    
    #\\\ ARCHITECTURE
    
    # Select architectural nn.Module to use
    modelAggGNN['archit'] = archit.AggregationGNN
    # Convolutional layers
    modelAggGNN['dimFeatures'] = [1, 16, 32] # Number of features per layer
    modelAggGNN['nFilterTaps'] = [4, 8] # Number of filter taps
    modelAggGNN['bias'] = True # Include bias
    # Nonlinearity
    modelAggGNN['nonlinearity'] = nn.ReLU
    # Pooling
    modelAggGNN['poolingFunction'] = nn.MaxPool1d # Summarizing function
    modelAggGNN['poolingSize'] = [2, 2] # Summarizing neighborhoods
    # Readout layer
    modelAggGNN['dimLayersMLP'] = [nClasses]
    # Graph structure
    modelAggGNN['GSO'] = None # To be determined later on, based on data
    modelAggGNN['order'] = None # To be determined next
    # Aggregation sequence
    modelAggGNN['maxN'] = None # Maximum number of exchanges
    modelAggGNN['nNodes'] = 1 # Number of nodes on which to obtain the 
        # aggregation sequence
    modelAggGNN['dimLayersAggMLP'] = [] # If more than one has been used, then
        # this MLP mixes together the features learned at all the selected nodes
        
    #\\\ TRAINER

    modelAggGNN['trainer'] = training.Trainer
    
    #\\\ EVALUATOR
    
    modelAggGNN['evaluator'] = evaluation.evaluate
        
#\\\\\\\\\\\\
#\\\ MODEL 5: Aggregation GNN with node selected by degree
#\\\\\\\\\\\\

if doAggregationGNN and doDegree:

    modelAggGNNdeg = deepcopy(modelAggGNN)

    modelAggGNNdeg['name'] += 'deg' # Name of the architecture
    # Structure
    modelAggGNNdeg['order'] = 'Degree'

    #\\\ Save Values:
    writeVarValues(varsFile, modelAggGNNdeg)
    modelList += [modelAggGNNdeg['name']]
    
#\\\\\\\\\\\\
#\\\ MODEL 6: Aggregation GNN with node selected by EDS
#\\\\\\\\\\\\

if doAggregationGNN and doEDS:

    modelAggGNNeds = deepcopy(modelAggGNN)

    modelAggGNNeds['name'] += 'eds' # Name of the architecture
    # Structure
    modelAggGNNeds['order'] = 'EDS'

    #\\\ Save Values:
    writeVarValues(varsFile, modelAggGNNeds)
    modelList += [modelAggGNNeds['name']]
    
#\\\\\\\\\\\\
#\\\ MODEL 7: Aggregation GNN with node selected by spectral proxies
#\\\\\\\\\\\\

if doAggregationGNN and doSpectralProxies:

    modelAggGNNspr = deepcopy(modelAggGNN)

    modelAggGNNspr['name'] += 'spr' # Name of the architecture
    # Structure
    modelAggGNNspr['order'] = 'SpectralProxies'

    #\\\ Save Values:
    writeVarValues(varsFile, modelAggGNNspr)
    modelList += [modelAggGNNspr['name']]


#\\\\\\\\\\\\\\\\\\\\\\\
#\\\\ MULTINODE GNN \\\\
#\\\\\\\\\\\\\\\\\\\\\\\

if doMultinodeGNN:
    #\\\ Basic parameters for all the Aggregation GNN architectures
    
    modelMultiGNN = {}
    
    modelMultiGNN['name'] = 'MultiGNN' # To be modified later on depending on the
        # specific ordering selected
    modelMultiGNN['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) \
                                     else 'cpu'
    
    #\\\ ARCHITECTURE
    
    # Select architectural nn.Module to use
    modelMultiGNN['archit'] = archit.MultiNodeAggregationGNN
    #Multinode params

    modelMultiGNN['nSelectedNodes'] = [10, 5]
    modelMultiGNN['nShifts'] = [7, 5]

    # Convolutional layers
    modelMultiGNN['dimFeatures'] = [[1, 16], [16, 32]] # Number of features per layer
    modelMultiGNN['nFilterTaps'] = [[2], [2]] # Number of filter taps
    modelMultiGNN['bias'] = True # Include bias
    # Nonlinearity
    modelMultiGNN['nonlinearity'] = nn.ReLU
    # Pooling
    modelMultiGNN['poolingFunction'] = nn.MaxPool1d # Summarizing function
    modelMultiGNN['poolingSize'] = [[2], [2]] # Summarizing neighborhoods
    # Readout layer
    modelMultiGNN['dimLayersMLP'] = [nClasses]
    # Graph structure
    modelMultiGNN['GSO'] = None # To be determined later on, based on data
    modelMultiGNN['order'] = None # To be determined next
        
    #\\\ TRAINER

    modelMultiGNN['trainer'] = training.Trainer
    
    #\\\ EVALUATOR
    
    modelMultiGNN['evaluator'] = evaluation.evaluate

        
#\\\\\\\\\\\\
#\\\ MODEL 8: Multinode GNN with node selected by degree
#\\\\\\\\\\\\

if doMultinodeGNN and doDegree:

    modelMultiGNNdeg = deepcopy(modelMultiGNN)

    modelMultiGNNdeg['name'] += 'deg' # Name of the architecture
    # Structure
    modelMultiGNNdeg['order'] = 'Degree'

    #\\\ Save Values:
    writeVarValues(varsFile, modelMultiGNNdeg)
    modelList += [modelMultiGNNdeg['name']]
    
#\\\\\\\\\\\\
#\\\ MODEL 9: Multinode GNN with node selected by EDS
#\\\\\\\\\\\\

if doMultinodeGNN and doEDS:

    modelMultiGNNeds = deepcopy(modelMultiGNN)

    modelMultiGNNeds['name'] += 'eds' # Name of the architecture
    # Structure
    modelMultiGNNeds['order'] = 'EDS'

    #\\\ Save Values:
    writeVarValues(varsFile, modelMultiGNNeds)
    modelList += [modelMultiGNNeds['name']]
    
#\\\\\\\\\\\\
#\\\ MODEL 10: Multinode GNN with node selected by spectral proxies
#\\\\\\\\\\\\

if doMultinodeGNN and doSpectralProxies:

    modelMultiGNNspr = deepcopy(modelMultiGNN)

    modelMultiGNNspr['name'] += 'spr' # Name of the architecture
    # Structure
    modelMultiGNNspr['order'] = 'SpectralProxies'

    #\\\ Save Values:
    writeVarValues(varsFile, modelMultiGNNspr)
    modelList += [modelMultiGNNspr['name']]


###########
# LOGGING #
###########

# Options:
doPrint = True # Decide whether to print stuff while running
doLogging = False # Log into tensorboard
doSaveVars = True # Save (pickle) useful variables
doFigs = True # Plot some figures (this only works if doSaveVars is True)
# Parameters:
printInterval = 100 # After how many training steps, print the partial results
xAxisMultiplierTrain = 100 # How many training steps in between those shown in
    # the plot, i.e., one training step every xAxisMultiplierTrain is shown.
xAxisMultiplierValid = 10 # How many validation steps in between those shown,
    # same as above.
figSize = 5 # Overall size of the figure that contains the plot
lineWidth = 1 # Width of the plot lines
markerShape = 'o' # Shape of the markers
markerSize = 2 # Size of the markers

#\\\ Save values:
writeVarValues(varsFile,
               {'doPrint': doPrint,
                'doLogging': doLogging,
                'doSaveVars': doSaveVars,
                'doFigs': doFigs,
                'saveDir': saveDir,
                'printInterval': printInterval,
                'figSize': figSize,
                'lineWidth': lineWidth,
                'markerShape': markerShape,
                'markerSize': markerSize})

#%%##################################################################
#                                                                   #
#                    SETUP                                          #
#                                                                   #
#####################################################################

#\\\ Determine processing unit:
if useGPU and torch.cuda.is_available():
    torch.cuda.empty_cache()

#\\\ Notify of processing units
if doPrint:
    print("Selected devices:")
    for thisModel in modelList:
        modelDict = eval('model' + thisModel)
        print("\t%s: %s" % (thisModel, modelDict['device']))

#\\\ Logging options
if doLogging:
    from alegnn.utils.visualTools import Visualizer
    logsTB = os.path.join(saveDir, 'logsTB')
    logger = Visualizer(logsTB, name='visualResults')
    
#\\\ Save variables during evaluation.
# We will save all the evaluations obtained for each of the trained models.
# It basically is a dictionary, containing a list. The key of the
# dictionary determines the model, then the first list index determines
# which split realization. Then, this will be converted to numpy to compute
# mean and standard deviation (across the split dimension).
costBest = {} # Cost for the best model (Evaluation cost: Error rate)
costLast = {} # Cost for the last model
for thisModel in modelList: # Create an element for each split realization,
    costBest[thisModel] = [None] * nGraphRealizations
    costLast[thisModel] = [None] * nGraphRealizations

if doFigs:
    #\\\ SAVE SPACE:
    # Create the variables to save all the realizations. This is, again, a
    # dictionary, where each key represents a model, and each model is a list
    # for each data split.
    # Each data split, in this case, is not a scalar, but a vector of
    # length the number of training steps (or of validation steps)
    lossTrain = {}
    costTrain = {}
    lossValid = {}
    costValid = {}
    # Initialize the splits dimension
    for thisModel in modelList:
        lossTrain[thisModel] = [None] * nGraphRealizations
        costTrain[thisModel] = [None] * nGraphRealizations
        lossValid[thisModel] = [None] * nGraphRealizations
        costValid[thisModel] = [None] * nGraphRealizations


####################
# TRAINING OPTIONS #
####################

# Training phase. It has a lot of options that are input through a
# dictionary of arguments.
# The value of this options was decided above with the rest of the parameters.
# This just creates a dictionary necessary to pass to the train function.

trainingOptions = {}

if doLogging:
    trainingOptions['logger'] = logger
if doSaveVars:
    trainingOptions['saveDir'] = saveDir
if doPrint:
    trainingOptions['printInterval'] = printInterval
if doLearningRateDecay:
    trainingOptions['learningRateDecayRate'] = learningRateDecayRate
    trainingOptions['learningRateDecayPeriod'] = learningRateDecayPeriod
trainingOptions['validationInterval'] = validationInterval

# And in case each model has specific training options, then we create a 
# separate dictionary per model.

trainingOptsPerModel= {}

#%%##################################################################
#                                                                   #
#                    GRAPH REALIZATION                              #
#                                                                   #
#####################################################################

# Start generating a new graph for each of the number of graph realizations that
# we previously specified.

# Unless it's the Facebook graph, which is fixed

# Load the graph and select the source nodes

if graphType == 'FacebookEgo':

    #########
    # GRAPH #
    #########

    if doPrint:
        print("Load data...", flush = True, end = ' ')

    # Create graph
    facebookData = alegnn.utils.dataTools.FacebookEgo(dataDir, use234)
    adjacencyMatrix = facebookData.getAdjacencyMatrix(use234)
    assert adjacencyMatrix is not None
    nNodes = adjacencyMatrix.shape[0]

    if doPrint:
        print("OK")
    # Now, to create the proper graph object, since we're going to use
    # 'fuseEdges' option in createGraph, we are going to add an extra dimension
    # to the adjacencyMatrix (to indicate there's only one matrix in the 
    # collection that we should be fusing)
    adjacencyMatrix = adjacencyMatrix.reshape([1, nNodes, nNodes])
    nodeList = []
    extraComponents = []
    if doPrint:
        print("Creating graph...", flush = True, end = ' ')
    graphOptions['adjacencyMatrices'] = adjacencyMatrix
    graphOptions['nodeList'] = nodeList
    graphOptions['extraComponents'] = extraComponents
    graphOptions['aggregationType'] = 'sum'
    graphOptions['normalizationType'] = 'no'
    graphOptions['forceUndirected'] = True
    
    G = graphTools.Graph('fuseEdges', nNodes, graphOptions)
    G.computeGFT() # Compute the eigendecomposition of the stored GSO
    
    nNodes = G.N
    
    if doPrint:
        print("OK")
    
    ################
    # SOURCE NODES #
    ################

    if doPrint:
        print("Selecting source nodes...", end = ' ', flush = True)
    # For the source localization problem, we have to select which ones, of all
    # the nodes, will act as source nodes. This is determined by a list of
    # indices indicating which nodes to choose as sources.
    sourceNodes = graphTools.computeSourceNodes(G.A, nClasses)
    if use234:
        sourceNodes = [38, 224]
    
    #\\\ Save values:
    writeVarValues(varsFile,
                   {'sourceNodes': sourceNodes})
    
    if doPrint:
        print("OK")

for graph in range(nGraphRealizations):

    # The accBest and accLast variables, for each model, have a list with a
    # total number of elements equal to the number of graphs we will generate
    # Now, for each graph, we have multiple data realization, so we want, for
    # each graph, to create a list to hold each of those values
    for thisModel in modelList:
        costBest[thisModel][graph] = []
        costLast[thisModel][graph] = []
        
        lossTrain[thisModel][graph] = []
        costTrain[thisModel][graph] = []
        lossValid[thisModel][graph] = []
        costValid[thisModel][graph] = []       

    #%%##################################################################
    #                                                                   #
    #                    DATA HANDLING                                  #
    #                                                                   #
    #####################################################################

    if graphType != 'FacebookEgo' and not graphType == "20news":
        # If the graph type is the Facebook one, then that graph is fixed,
        # so we don't have to keep changing it.

        #########
        # GRAPH #
        #########

        # Create graph
        G = graphTools.Graph(graphType, nNodes, graphOptions)
        G.computeGFT() # Compute the eigendecomposition of the stored GSO

        ################
        # SOURCE NODES #
        ################

        # For the source localization problem, we have to select which ones, of
        # all the nodes, will act as source nodes. This is determined by a list
        # of indices indicating which nodes to choose as sources.
        sourceNodes = graphTools.computeSourceNodes(G.A, nClasses)
        
        #\\\ Save values:
        writeVarValues(varsFile,
                       {'sourceNodes': sourceNodes})

    # We have now created the graph and selected the source nodes on that graph.
    # So now we proceed to generate random data realizations, different
    # realizations of diffusion processes.

    for realization in range(nDataRealizations):

        

        #%%##################################################################
        #                                                                   #
        #                    MODELS INITIALIZATION                          #
        #                                                                   #
        #####################################################################
        
        # This is the dictionary where we store the models (in a model.Model
        # class, that is then passed to training).
        modelsGNN = {}
    
        # If a new model is to be created, it should be called for here.
        
        if doPrint:
            print("Model initialization...", flush = True)
            
        for thisModel in modelList:
            
            # Get the corresponding parameter dictionary
            modelDict = deepcopy(eval('model' + thisModel))
            # and training options
            trainingOptsPerModel[thisModel] = deepcopy(trainingOptions)
            
            # Now, this dictionary has all the hyperparameters that we need to
            # pass to the architecture function, but it also has other keys
            # that belong to the more general model (like 'name' or 'device'),
            # so we need to extract them and save them in seperate variables
            # for future use.
            thisName = modelDict.pop('name')
            callArchit = modelDict.pop('archit')
            thisDevice = modelDict.pop('device')
            thisTrainer = modelDict.pop('trainer')
            thisEvaluator = modelDict.pop('evaluator')
            
            # If more than one graph or data realization is going to be 
            # carried out, we are going to store all of thos models
            # separately, so that any of them can be brought back and
            # studied in detail.
            if nGraphRealizations > 1:
                thisName += 'G%02d' % graph
            if nDataRealizations > 1:
                thisName += 'R%02d' % realization
                
            if doPrint:
                print("\tInitializing %s..." % thisName,
                      end = ' ',flush = True)
                
            ##############
            # PARAMETERS #
            ##############
    
            #\\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisOptimAlg = optimAlg
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2
    
            #\\\ GSO
            # The coarsening technique is defined for the normalized and
            # rescaled Laplacian, whereas for the other ones we use the
            # normalized adjacency
            L = graphTools.normalizeLaplacian(G.L)
            EL, VL = graphTools.computeGFT(L, order ='increasing')
            S = 2*L/np.max(np.real(EL)) - np.eye(nNodes)
        
            modelDict['GSO'] = S
            
            ################
            # ARCHITECTURE #
            ################
    
            thisArchit = callArchit(**modelDict)
            
            #############
            # OPTIMIZER #
            #############
    
            if thisOptimAlg == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr = learningRate,
                                       betas = (beta1, beta2))
            elif thisOptimAlg == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(),
                                      lr = learningRate)
            elif thisOptimAlg == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr = learningRate, alpha = beta1)
    
            ########
            # LOSS #
            ########
    
            # Initialize the loss function
            thisLossFunction = loss.adaptExtraDimensionLoss(lossFunction)
            if 'GLGNN' in thisModel:
                
                thisLossFunction = loss.MultiGraphLearningLoss(lossFunction, 
                                thisArchit.S, 
                                thisArchit.signals,
                                thisArchit.enc_dec_errors,
                                multipliers)
                thisOptim = GLOptim.MultiGraphLearningOptimizer(thisArchit.named_parameters(), 
                                                thisArchit.constants,
                                                lr = lr,
                                                betas = (beta1,beta2),
                                                weight_decay=2e-5)
            #########
            # MODEL #
            #########
    
            # Create the model
            modelCreated = model.Model(thisArchit,
                                       thisLossFunction,
                                       thisOptim,
                                       thisTrainer,
                                       thisEvaluator,
                                       thisDevice,
                                       thisName,
                                       saveDir)
    
            # Store it
            modelsGNN[thisName] = modelCreated
            # Write the main hyperparameters
            writeVarValues(varsFile,
                           {'name': thisName,
                            'thisOptimizationAlgorithm': thisOptimAlg,
                            'thisTrainer': thisTrainer,
                            'thisEvaluator': thisEvaluator,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})
    
            if doPrint:
                print("OK")
                
        if doPrint:
            print("Model initialization... COMPLETE")
    
        #%%##################################################################
        #                                                                   #
        #                    TRAINING                                       #
        #                                                                   #
        #####################################################################
        print("")
        
        # We train each model separately
    
        for thisModel in modelsGNN.keys():
            
            if doPrint:
                print("Training model %s..." % thisModel)
             
            # Remember that modelsGNN.keys() has the split numbering as well as 
            # the name, while modelList has only the name. So we need to map 
            # the specific model for this specific split with the actual model
            # name, since there are several variables that are indexed by the
            # model name (for instance, the training options, or the
            # dictionaries saving the loss values)
            for m in modelList:
                if m in thisModel:
                    modelName = m
        
            # Identify the specific graph and data realizations at training time
            if nGraphRealizations > 1:
                trainingOptions['graphNo'] = graph
            if nDataRealizations > 1:
                trainingOptions['realizationNo'] = realization
            
            # Train the model
            thisTrainVars = modelsGNN[thisModel].train(data,
                                                       nEpochs,
                                                       batchSize,
                                                       **trainingOptsPerModel[modelName])
    
            if doFigs:
            # Find which model to save the results (when having multiple
            # realizations)
                lossTrain[modelName][graph] += [thisTrainVars['lossTrain']]
                costTrain[modelName][graph] += [thisTrainVars['costTrain']]
                lossValid[modelName][graph] += [thisTrainVars['lossValid']]
                costValid[modelName][graph] += [thisTrainVars['costValid']]
                        
        # And we also need to save 'nBatch' but is the same for all models, so
        if doFigs:
            nBatches = thisTrainVars['nBatches']

        #%%##################################################################
        #                                                                   #
        #                    EVALUATION                                     #
        #                                                                   #
        #####################################################################
        
        # Now that the model has been trained, we evaluate them on the test
        # samples.
    
        # We have two versions of each model to evaluate: the one obtained
        # at the best result of the validation step, and the last trained model.
    
        if doPrint:
            print("\nTotal testing error rate", end = '', flush = True)
            if nGraphRealizations > 1 or nDataRealizations > 1:
                print(" (", end = '', flush = True)
                if nGraphRealizations > 1:
                    print("Graph %02d" % graph, end = '', flush = True)
                    if nDataRealizations > 1:
                        print(", ", end = '', flush = True)
                if nDataRealizations > 1:
                    print("Realization %02d" % realization, end = '',
                          flush = True)
                print(")", end = '', flush = True)
            print(":", flush = True)
            
    
        for thisModel in modelsGNN.keys():
            
            # Same as before, separate the model name from the data or graph
            # realization number
            for m in modelList:
                if m in thisModel:
                    modelName = m
    
            # Evaluate the model
            thisEvalVars = modelsGNN[thisModel].evaluate(data)
            
            # Save the outputs
            thisCostBest = thisEvalVars['costBest']
            thisCostLast = thisEvalVars['costLast']
            
            # Write values
            writeVarValues(varsFile,
                           {'costBest%s' % thisModel: thisCostBest,
                            'costLast%s' % thisModel: thisCostLast})
    
            # Now check which is the model being trained
            costBest[modelName][graph] += [thisCostBest]
            costLast[modelName][graph] += [thisCostLast]
            # This is so that we can later compute a total accuracy with
            # the corresponding error.
            
            if doPrint:
                print("\t%s: %6.2f%% [Best] %6.2f%% [Last]" % (thisModel,
                                                               thisCostBest*100,
                                                               thisCostLast*100))

############################
# FINAL EVALUATION RESULTS #
############################

# Now that we have computed the accuracy of all runs, we can obtain a final
# result (mean and standard deviation)

meanCostBestPerGraph = {} # Compute the mean accuracy (best) across all
    # realizations data realizations of a graph
meanCostLastPerGraph = {} # Compute the mean accuracy (last) across all
    # realizations data realizations of a graph
meanCostBest = {} # Mean across graphs (after having averaged across data
    # realizations)
meanCostLast = {} # Mean across graphs
stdDevCostBest = {} # Standard deviation across graphs
stdDevCostLast = {} # Standard deviation across graphs

if doPrint:
    print("\nFinal evaluations (%02d graphs, %02d realizations)" % (
            nGraphRealizations, nDataRealizations))

for thisModel in modelList:
    # Convert the lists into a nGraphRealizations x nDataRealizations matrix
    costBest[thisModel] = np.array(costBest[thisModel])
    costLast[thisModel] = np.array(costLast[thisModel])
    
    if nGraphRealizations == 1 or nDataRealizations == 1:
        meanCostBestPerGraph[thisModel] = np.squeeze(costBest[thisModel])
        meanCostLastPerGraph[thisModel] = np.squeeze(costLast[thisModel])
    else:
        # Compute the mean (across realizations for a given graph)
        meanCostBestPerGraph[thisModel] = np.mean(costBest[thisModel], axis = 1)
        meanCostLastPerGraph[thisModel] = np.mean(costLast[thisModel], axis = 1)

    # And now compute the statistics (across graphs)
    meanCostBest[thisModel] = np.mean(meanCostBestPerGraph[thisModel])
    meanCostLast[thisModel] = np.mean(meanCostLastPerGraph[thisModel])
    stdDevCostBest[thisModel] = np.std(meanCostBestPerGraph[thisModel])
    stdDevCostLast[thisModel] = np.std(meanCostLastPerGraph[thisModel])

    # And print it:
    if doPrint:
        print("\t%s: %6.2f%% (+-%6.2f%%) [Best] %6.2f%% (+-%6.2f%%) [Last]" % (
                thisModel,
                meanCostBest[thisModel] * 100,
                stdDevCostBest[thisModel] * 100,
                meanCostLast[thisModel] * 100,
                stdDevCostLast[thisModel] * 100))

    # Save values
    writeVarValues(varsFile,
               {'meanCostBest%s' % thisModel: meanCostBest[thisModel],
                'stdDevCostBest%s' % thisModel: stdDevCostBest[thisModel],
                'meanCostLast%s' % thisModel: meanCostLast[thisModel],
                'stdDevCostLast%s' % thisModel : stdDevCostLast[thisModel]})
    
with open(varsFile, 'a+') as file:
    file.write("Final evaluations (%02d graphs, %02d realizations)\n" % (
            nGraphRealizations, nDataRealizations))
    for thisModel in modelList:
        file.write("\t%s: %6.2f%% (+-%6.2f%%) [Best] %6.2f%% (+-%6.2f%%) [Last]\n" % (
                   thisModel,
                   meanCostBest[thisModel] * 100,
                   stdDevCostBest[thisModel] * 100,
                   meanCostLast[thisModel] * 100,
                   stdDevCostLast[thisModel] * 100))
    file.write('\n')

#%%##################################################################
#                                                                   #
#                    PLOT                                           #
#                                                                   #
#####################################################################

# Finally, we might want to plot several quantities of interest

if doFigs and doSaveVars:
    
    ###################
    # DATA PROCESSING #
    ###################
    
    #\\\ FIGURES DIRECTORY:
    saveDirFigs = os.path.join(saveDir,'figs')
    # If it doesn't exist, create it.
    if not os.path.exists(saveDirFigs):
        os.makedirs(saveDirFigs)

    #\\\ COMPUTE STATISTICS:
    # The first thing to do is to transform those into a matrix with all the
    # realizations, so create the variables to save that.
    meanLossTrainPerGraph = {}
    meanCostTrainPerGraph = {}
    meanLossValidPerGraph = {}
    meanCostValidPerGraph = {}
    meanLossTrain = {}
    meanCostTrain = {}
    meanLossValid = {}
    meanCostValid = {}
    stdDevLossTrain = {}
    stdDevCostTrain = {}
    stdDevLossValid = {}
    stdDevCostValid = {}
    # Initialize the variables
    for thisModel in modelList:
        meanLossTrainPerGraph[thisModel] = [None] * nGraphRealizations
        meanCostTrainPerGraph[thisModel] = [None] * nGraphRealizations
        meanLossValidPerGraph[thisModel] = [None] * nGraphRealizations
        meanCostValidPerGraph[thisModel] = [None] * nGraphRealizations
        if nGraphRealizations > 1:
            for G in range(nGraphRealizations):
                # Transform into np.array
                lossTrain[thisModel][G] = np.array(lossTrain[thisModel][G])
                costTrain[thisModel][G] = np.array(costTrain[thisModel][G])
                lossValid[thisModel][G] = np.array(lossValid[thisModel][G])
                costValid[thisModel][G] = np.array(costValid[thisModel][G])
                # So, finally, for each model and each graph, we have a np.array of
                # shape:  nDataRealizations x number_of_training_steps
                # And we have to average these to get the mean across all data
                # realizations for each graph
                meanLossTrainPerGraph[thisModel][G] = \
                                    np.mean(lossTrain[thisModel][G], axis = 0)
                meanCostTrainPerGraph[thisModel][G] = \
                                    np.mean(costTrain[thisModel][G], axis = 0)
                meanLossValidPerGraph[thisModel][G] = \
                                    np.mean(lossValid[thisModel][G], axis = 0)
                meanCostValidPerGraph[thisModel][G] = \
                                    np.mean(costValid[thisModel][G], axis = 0)
        else:
            meanLossTrainPerGraph[thisModel] = lossTrain[thisModel][0]
            meanCostTrainPerGraph[thisModel] = costTrain[thisModel][0]
            meanLossValidPerGraph[thisModel] = lossValid[thisModel][0]
            meanCostValidPerGraph[thisModel] = costValid[thisModel][0]
        # And then convert this into np.array for all graphs
        meanLossTrainPerGraph[thisModel] = \
                                    np.array(meanLossTrainPerGraph[thisModel])
        meanCostTrainPerGraph[thisModel] = \
                                    np.array(meanCostTrainPerGraph[thisModel])
        meanLossValidPerGraph[thisModel] = \
                                    np.array(meanLossValidPerGraph[thisModel])
        meanCostValidPerGraph[thisModel] = \
                                    np.array(meanCostValidPerGraph[thisModel])
        # And compute the statistics
        meanLossTrain[thisModel] = \
                            np.mean(meanLossTrainPerGraph[thisModel], axis = 0)
        meanCostTrain[thisModel] = \
                            np.mean(meanCostTrainPerGraph[thisModel], axis = 0)
        meanLossValid[thisModel] = \
                            np.mean(meanLossValidPerGraph[thisModel], axis = 0)
        meanCostValid[thisModel] = \
                            np.mean(meanCostValidPerGraph[thisModel], axis = 0)
        stdDevLossTrain[thisModel] = \
                            np.std(meanLossTrainPerGraph[thisModel], axis = 0)
        stdDevCostTrain[thisModel] = \
                            np.std(meanCostTrainPerGraph[thisModel], axis = 0)
        stdDevLossValid[thisModel] = \
                            np.std(meanLossValidPerGraph[thisModel], axis = 0)
        stdDevCostValid[thisModel] = \
                            np.std(meanCostValidPerGraph[thisModel], axis = 0)

    ####################
    # SAVE FIGURE DATA #
    ####################

    # And finally, we can plot. But before, let's save the variables mean and
    # stdDev so, if we don't like the plot, we can re-open them, and re-plot
    # them, a piacere.
    varsPickle = {}
    varsPickle['nEpochs'] = nEpochs
    varsPickle['nBatches'] = nBatches
    varsPickle['meanLossTrain'] = meanLossTrain
    varsPickle['stdDevLossTrain'] = stdDevLossTrain
    varsPickle['meanCostTrain'] = meanCostTrain
    varsPickle['stdDevCostTrain'] = stdDevCostTrain
    varsPickle['meanLossValid'] = meanLossValid
    varsPickle['stdDevLossValid'] = stdDevLossValid
    varsPickle['meanCostValid'] = meanCostValid
    varsPickle['stdDevCostValid'] = stdDevCostValid
    with open(os.path.join(saveDirFigs,'figVars.pkl'), 'wb') as figVarsFile:
        pickle.dump(varsPickle, figVarsFile)

    ########
    # PLOT #
    ########

    # Compute the x-axis
    xTrain = np.arange(0, nEpochs * nBatches, xAxisMultiplierTrain)
    xValid = np.arange(0, nEpochs * nBatches, \
                          validationInterval*xAxisMultiplierValid)

    # If we do not want to plot all the elements (to avoid overcrowded plots)
    # we need to recompute the x axis and take those elements corresponding
    # to the training steps we want to plot
    if xAxisMultiplierTrain > 1:
        # Actual selected samples
        selectSamplesTrain = xTrain
        # Go and fetch tem
        for thisModel in modelList:
            meanLossTrain[thisModel] = meanLossTrain[thisModel]\
                                                    [selectSamplesTrain]
            stdDevLossTrain[thisModel] = stdDevLossTrain[thisModel]\
                                                        [selectSamplesTrain]
            meanCostTrain[thisModel] = meanCostTrain[thisModel]\
                                                    [selectSamplesTrain]
            stdDevCostTrain[thisModel] = stdDevCostTrain[thisModel]\
                                                        [selectSamplesTrain]
    # And same for the validation, if necessary.
    if xAxisMultiplierValid > 1:
        selectSamplesValid = np.arange(0, len(meanLossValid[thisModel]), \
                                       xAxisMultiplierValid)
        for thisModel in modelList:
            meanLossValid[thisModel] = meanLossValid[thisModel]\
                                                    [selectSamplesValid]
            stdDevLossValid[thisModel] = stdDevLossValid[thisModel]\
                                                        [selectSamplesValid]
            meanCostValid[thisModel] = meanCostValid[thisModel]\
                                                    [selectSamplesValid]
            stdDevCostValid[thisModel] = stdDevCostValid[thisModel]\
                                                        [selectSamplesValid]

    #\\\ LOSS (Training and validation) for EACH MODEL
    for key in meanLossTrain.keys():
        lossFig = plt.figure(figsize=(1.61*figSize, 1*figSize))
        plt.errorbar(xTrain, meanLossTrain[key], yerr = stdDevLossTrain[key],
                     color = '#01256E', linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
        plt.errorbar(xValid, meanLossValid[key], yerr = stdDevLossValid[key],
                     color = '#95001A', linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
        plt.ylabel(r'Loss')
        plt.xlabel(r'Training steps')
        plt.legend([r'Training', r'Validation'])
        plt.title(r'%s' % key)
        lossFig.savefig(os.path.join(saveDirFigs,'loss%s.pdf' % key),
                        bbox_inches = 'tight')

    #\\\ RMSE (Training and validation) for EACH MODEL
    for key in meanCostTrain.keys():
        costFig = plt.figure(figsize=(1.61*figSize, 1*figSize))
        plt.errorbar(xTrain, meanCostTrain[key], yerr = stdDevCostTrain[key],
                     color = '#01256E', linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
        plt.errorbar(xValid, meanCostValid[key], yerr = stdDevCostValid[key],
                     color = '#95001A', linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
        plt.ylabel(r'Error rate')
        plt.xlabel(r'Training steps')
        plt.legend([r'Training', r'Validation'])
        plt.title(r'%s' % key)
        costFig.savefig(os.path.join(saveDirFigs,'cost%s.pdf' % key),
                        bbox_inches = 'tight')

    # LOSS (training) for ALL MODELS
    allLossTrain = plt.figure(figsize=(1.61*figSize, 1*figSize))
    for key in meanLossTrain.keys():
        plt.errorbar(xTrain, meanLossTrain[key], yerr = stdDevLossTrain[key],
                     linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
    plt.ylabel(r'Loss')
    plt.xlabel(r'Training steps')
    plt.legend(list(meanLossTrain.keys()))
    allLossTrain.savefig(os.path.join(saveDirFigs,'allLossTrain.pdf'),
                    bbox_inches = 'tight')

    # RMSE (validation) for ALL MODELS
    allCostValidFig = plt.figure(figsize=(1.61*figSize, 1*figSize))
    for key in meanCostValid.keys():
        plt.errorbar(xValid, meanCostValid[key], yerr = stdDevCostValid[key],
                     linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
    plt.ylabel(r'Error rate')
    plt.xlabel(r'Training steps')
    plt.legend(list(meanCostValid.keys()))
    allCostValidFig.savefig(os.path.join(saveDirFigs,'allCostValid.pdf'),
                    bbox_inches = 'tight')

# Finish measuring time
endRunTime = datetime.datetime.now()

totalRunTime = abs(endRunTime - startRunTime)
totalRunTimeH = int(divmod(totalRunTime.total_seconds(), 3600)[0])
totalRunTimeM, totalRunTimeS = \
               divmod(totalRunTime.total_seconds() - totalRunTimeH * 3600., 60)
totalRunTimeM = int(totalRunTimeM)

if doPrint:
    print(" ")
    print("Simulation started: %s" %startRunTime.strftime("%Y/%m/%d %H:%M:%S"))
    print("Simulation ended:   %s" % endRunTime.strftime("%Y/%m/%d %H:%M:%S"))
    print("Total time: %dh %dm %.2fs" % (totalRunTimeH,
                                         totalRunTimeM,
                                         totalRunTimeS))
    
# And save this info into the .txt file as well
with open(varsFile, 'a+') as file:
    file.write("Simulation started: %s\n" % 
                                     startRunTime.strftime("%Y/%m/%d %H:%M:%S"))
    file.write("Simulation ended:   %s\n" % 
                                       endRunTime.strftime("%Y/%m/%d %H:%M:%S"))
    file.write("Total time: %dh %dm %.2fs" % (totalRunTimeH,
                                              totalRunTimeM,
                                              totalRunTimeS))
