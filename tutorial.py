#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import numpy as np
import pickle
import datetime
from copy import deepcopy


# Next, we import <code>matplotlib</code> to be able to plot the loss and evaluation measures after training. Note that <code>matplotlib</code> is configured to use LaTeX, so a corresponding LaTeX installation is required. If not, please comment the appropriate lines.

# In[2]:


import matplotlib
matplotlib.rcParams['text.usetex'] = False # Comment this line if no LaTeX installation is available
#matplotlib.rcParams['font.family'] = 'serif' # Comment this line if no LaTeX installation is available
#matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# The training of the neural network model runs on [PyTorch](https://pytorch.org/get-started/locally/), so we need to import it, together with the two libraries that will be used frequently, <code>torch.nn</code> and <code>torch.optim</code> that are imported with shortcuts.

# In[3]:


import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim


# Finally, we import the core libraries that have all the required classes and functions to run the graph neural networks.
# 
# The library <code>Utils.graphTools</code> has all the basic functions to handle and operate on graphs, together with a <code>Graph</code> class that binds together different graph representations and methods. The library <code>Utils.dataTools</code> has the corresponding classes for the different datasets, in particular in this tutorial, we will focus on the class <code>SourceLocalization</code>. The last utility library, called <code>Utils.graphML</code> contains all the graph neural network layers and functions. More specifically, <code>Utils.graphML</code> attempts to mimic <code>torch.nn</code> by defining only the graph neural network layers as <code>nn.Module</code> classes (including the linear layers, the activation functions, and the pooling layers) as well as the corresponding functionals (akin to <code>torch.nn.functional</code>).
# 
# Next, the library <code>Modules.architectures</code> has some pre-specified GNN architectures (Selection GNN, Aggregation GNN, Spectral GNN, Graph Attention Networks, etc.), that are built from the layers provided in <code>Utils.graphML</code>. The library <code>Modules.model</code> contains a <code>Model</code> class that binds together the three main elements of each neural network model: the architecture, the loss function and the optimizer. It also determines the trainer and the evaluator and has, correspondingly, built-in methods for training and evaluation. It also offers other utilities such as saving and loading models, as well as individual training. The libraries <code>Modules.training</code>, <code>Modules.evaluation</code> and <code>Modules.loss</code> contains different classes and functions that set the measures of performance and the specifics of training. The library <code>Utils.miscTools</code> contains several miscellaneous tools, of which we care about <code>writeVarValues</code> which writes some desired values in a <code>.txt</code> file, and <code>saveSeed</code> which saves both the <code>numpy</code> and <code>torch</code> seeds for reproducibility.

# In[4]:


import alegnn.utils.graphTools as graphTools
import alegnn.utils.dataTools
import alegnn.utils.graphML as gml


# (A deprecation warning from the package <code>hdf5storage</code> might arise; this package is used to load the data for the [authorship attribution](https://ieeexplore.ieee.org/document/6638728) dataset)

# In[5]:


import alegnn.modules.architectures as archit
import alegnn.modules.model as model
import alegnn.modules.training as training
import alegnn.modules.evaluation as evaluation
import alegnn.modules.loss as loss


# In[6]:


from alegnn.utils.miscTools import writeVarValues
from alegnn.utils.miscTools import saveSeed


# ## Simulation Parameters <a class="anchor" id="sec:simulationParameters"></a>
# We define the basic simulation parameters: file handling, dataset generation, hyperparameter definition.

# ### File handling <a class="anchor" id="subsec:fileHandling"></a>
# Create a directory where to save the run. The directory will have the name determined by <code>thisFilename</code> variable and the type of graph, as well as the date and time of the run.

# In[7]:


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


# Next, we create a <code>.txt</code> file where we will save all of these parameters, so we know exactly how the run was called.

# In[8]:


# Create the file where all the (hyper)parameters are results will be saved.
varsFile = os.path.join(saveDir,'hyperparameters.txt')
with open(varsFile, 'w+') as file:
    file.write('%s\n\n' % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))


# Now, decide if we are going to use the GPU or not.

# In[9]:


useGPU = True


# Finally, we save the seeds of both <code>numpy</code> and <code>torch</code> to facilitate reproducibility. The <code>saveSeed</code> function in <code>Utils.miscTools</code> requires a <code>list</code> where each element of the list saves a <code>dict</code>ionary containing the <code>'module'</code> name (i.e. 'numpy' or 'torch' as a <code>string</code>) and the random number generator states and/or seed.

# In[10]:


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


# ### Data parameters <a class="anchor" id="subsec:dataParameters"></a>
# Next, we define the parameters for [generating the data](#subsec:sourceLoc).
# 
# First, the number of training, validation and test samples.

# In[11]:


nTrain = 5000 # Number of training samples
nValid = int(0.2 * nTrain) # Number of validation samples
nTest = 50 # Number of testing samples


# Then, the number of nodes and the number of communities of the stochastic block model graph (that was selected before when defining <code>graphType = 'SBM'</code>). Recall that the objective of the problem is to determine which community originated the diffusion, and as such, the number of communities is equal to the number of classes.

# In[12]:


nNodes = 20 # Number of nodes
nClasses = 2 # Number of classes (i.e. number of communities)
graphOptions = {} # Dictionary of options to pass to the graphTools.createGraph function
graphOptions['nCommunities'] = nClasses # Number of communities
graphOptions['probIntra'] = 0.8 # Probability of drawing edges intra communities
graphOptions['probInter'] = 0.2 # Probability of drawing edges inter communities


# Finally, we need to determine the maximum value for which the diffusion can run for. [Recall](#subsec:sourceLoc) that each input sample has the form
# 
# $$ \mathbf{x} = \mathbf{W}^{t} \boldsymbol{\delta}_{c} $$
# 
# where $\mathbf{x}$ is the graph signal, $\mathbf{W}$ is the adjacency matrix and $\boldsymbol{\delta}_{c}$ is the delta signal for community $c$ (i.e. a graph signal that has $0$ in every element, except for a $1$ in the source element of community $c$). The value of $t$ determines for <em>how long</em> the $\boldsymbol{\delta}_{c}$ has been diffusing. In the simulations, we pick the value of $t$ at random, for each sample, between $0$ and $t_{\max}$. That value of $t_{\max}$ is defined next. (Note that setting <code>tMax = None</code> is equivalent to setting <code>tMax = nNodes - 1</code> so that any diffusion length can appear in the generated dataset. For large graphs it might be convenient to limit the number of <code>tMax</code> for numerical stability.)

# In[13]:


tMax = None # Maximum number of diffusion times (W^t for t < tMax)


# Save all these values into the <code>.txt</code> file.

# In[14]:


#\\\ Save values:
writeVarValues(varsFile, {'nNodes': nNodes, 'graphType': graphType})
writeVarValues(varsFile, graphOptions)
writeVarValues(varsFile, {'nTrain': nTest,
                          'nValid': nValid,
                          'nTest': nTest,
                          'tMax': tMax,
                          'nClasses': nClasses,
                          'useGPU': useGPU})


# ### Training parameters <a class="anchor" id="subsec:trainingParameters"></a>
# The parameters for training the graph neural network are defined next.

# First, we determine de loss function we will use. In our case, the cross entropy loss, since this is a classification problem (recall that the cross entropy loss applies a softmax before feeding it into the negative log-likelihood, so there is no need to apply a softmax after the last layer of the graph neural network). Also, note that we do not need to initialize the loss function, since this will be initialized for each model separately.

# In[15]:

multipliers = {"lambda" : [0.0, 0.0],
               "gamma" : [0.0, 0.0],
               "beta" : [0.0, 0.0]}
lossFunction = nn.CrossEntropyLoss
customLoss = loss.MultiGraphLearningLoss



# Now that we have selected the loss function, we need to determine how to handle the training and evaluation. This, mostly, amounts to selecting wrappers that will handle the batch size partitioning, early stopping, validation, etc. The specifics of the evaluation measure, for example, depend on the data being measured and, thus, are parte of the <code>data</code> class.

# In[16]:


trainer = training.Trainer
evaluator = evaluation.evaluate


# Next, we determine the optimizer we use with all its parameters. In our case, an ADAM optimizer, where the variables <code>beta1</code> and <code>beta2</code> are the forgetting factors $\beta_{1}$ and $\beta_{2}$.

# In[17]:


optimAlg = 'ADAM'
learningRate = 0.001
beta1 = 0.9
beta2 = 0.999


# Finally, we determine the training process. The number of epochs, the batch size, and how often we carry out validation (i.e. after how many update steps, we run a validation step).

# In[18]:


nEpochs = 40 # Number of epochs
batchSize = 20 # Batch size
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


# ### Architecture hyperparameters <a class="anchor" id="subsec:architHyperparameters"></a>
# Now, we determine the architecture hyperparameters for the three architectures we will test: Aggregation GNN, Selection GNN with zero-padding, and Selection GNN with graph coarsening. We create a list to save all these models.

# In[20]:


modelList = []


# #### Aggregation GNN <a class="anchor" id="subsubsec:AggGNNhyper"></a>

# Let's start with the AggregationGNN. [Recall](#subsubsec:AggGNN) that, given the graph GSO $\mathbf{S}$ and the $f$th feature of the input graph signal $\mathbf{x}^{f}$, with $f=1,\ldots,F$ input features, we build the aggregation sequence at node $i \in \mathcal{V}$.
# 
# $$ \mathbf{z}_{i}^{f} = \big[ [\mathbf{x}^{f}]_{i}, [\mathbf{S} \mathbf{x}^{f}]_{i}, \ldots, [\mathbf{S}^{N_{\max}-1} \mathbf{x}^{f}]_{i} \big] $$
# 
# The first two elements to determine, then, are how many nodes we are going to aggregate this structure at, and how many exchanges we are going to do to build the sequence. (Note that we save all the <code>h</code>yper<code>Param</code>eter<code>s</code> of the <code>Agg</code>regation <code>GNN</code> in a dictionary). We select only one node, and we set $N_{\max}$ to <code>None</code> so that the number of exchanges is
# equal to the size of the network (guaranteeing that no information is lost when building the aggregation sequence $\mathbf{z}_{i}^{f}$).

# In[21]:


hParamsAggGNN = {}
hParamsAggGNN['name'] = 'AggGNN' # We give a name to this architecture

hParamsAggGNN['nNodes'] = 1 # The nodes are selected starting from the 
    # top of the signal vector, for the order given in the data. Later
    # we reorder the data to follow the highest-degree criteria.
hParamsAggGNN['Nmax'] = None # If 'None' sets maxN equal to the size
    # of the graph, so that no information is lost when creating the
    # aggregation sequence z_{i}


# The node(s) selected is (are) determined by some criteria that is chosen by the user. In this case, we will select it based on their degree (i.e. we pick the node(s) with the largest degree). To specify this, we determine another design choice named <code>order</code>. The name comes from the fact that the algorithm selects always the nodes from the top of the vector, so we need to reorder the elements in the vector (permute the nodes) so that the one on top is the one with largest degree. This achieved by several permutation functions available in <code>Utils.graphTools</code>. Right now there are three different criteria: by degree (<code>Degree</code>), by their <a href="https://ieeexplore.ieee.org/abstract/document/7383741" target="_blank">experimentally design sampling</a> score (<code>EDS</code>) or by their <a href="https://ieeexplore.ieee.org/document/7439829" target="_blank">spectral proxies</a> score (<code>SpectralProxies</code>). We note that any other criteria can be added by creating a function called <code>Perm</code> followed by the name of the method (for instance, <code>permDegree</code>) and this has to be specified here by whatever follows the word <code>perm</code>. This function is expected to take the graph matrix $\mathbf{S}$ and return the permuted matrix $\mathbf{\hat{S}}$ as well as a vector containing the ordering map.

# In[22]:


hParamsAggGNN['order'] = 'Degree'


# Now that we have set the hyperparameters to build the aggregation sequence $\mathbf{z}_{i}^{f}$ we [recall](#subsubsec:AggGNN) that this sequence offers a regular structure, since consecutive elements of this vector represent information from consecutive neighborhoods in the graph. Thus, if we have a regular structure, we can go ahead and apply a regular convolutional layer, and regular pooling.
# 
# Next, we define how many features <code>F</code> we want as the output of each layer, and how many filter taps <code>K</code> we use. These are determined by two lists, one for the features, and one for the filter taps. The features list has to have one more element, since it is the number of input features (i.e. the value of $F$ of the input $\mathbf{x}^{f}$ for $f=1,\ldots,F$). We decide for a two-layer GNN with $5$ output features on each layer, and $3$ filter taps on each layer. The number of input features is $F=1$ since each sample we use as input is simply the diffusion of a graph signal $\mathbf{x}^{1} = \mathbf{x} = \mathbf{W}^{t} \boldsymbol{\delta}_{c}$.

# In[23]:


hParamsAggGNN['F'] = [1, 5, 5] # Features per layer (the first element is the number of input features)
hParamsAggGNN['K'] = [3, 3] # Number of filter taps per layer
hParamsAggGNN['bias'] = True # Decide whether to include a bias term


# For the nonlinearity $\sigma$, after the filtering layer, we choose a ReLU. For the pooling function $\rho$, we choose a max-pooling, encompassing a number of elements given by $\alpha$ for each layer (i.e. how many elements of the vector to pool together). We choose $\alpha_{1}=2$ for the output fo the first layer (we compute the maximum of every $2$ elements in the vector obtained after applying the first convolutional layer followed by the nonlinearity), and $\alpha_{2} = 3$ for the output of the second layer.
# 
# We note that, since the aggregation sequence $\mathbf{z}_{i}^{f}$ exhibits a regular structure, then we just need to apply a pointwise nonlinearity and a regular pooling. As such, these functions are already efficiently implemented in the corresponding PyTorch library <code>torch.nn</code> so we simply point to them.

# In[24]:


hParamsAggGNN['sigma'] = nn.ReLU # Selected nonlinearity
hParamsAggGNN['rho'] = nn.MaxPool1d # Pooling function
hParamsAggGNN['alpha'] = [2, 3] # Size of pooling function


# Finally, once we have determined the convolutional layers, with their nonlinearities and pooling, we apply a simple one-layer MLP (i.e. a fully connected layer) to adapt the output dimension to have a total number of features equal to the number of classes (if we want to apply a deeper MLP, we add more elements to the list, each element determining how many output features after each fully connected layer, and we note that the nonlinearity applied between the layers is the same determined before; the last layer has no nonlinearity applied since the softmax is applied by the loss function).

# In[25]:


hParamsAggGNN['dimLayersMLP'] = [nClasses]


# We save this hyperparameters to the <code>.txt</code> file and add the architecture to the model list.

# In[26]:


writeVarValues(varsFile, hParamsAggGNN)
modelList += [hParamsAggGNN['name']]


# Note that if we select more than one node to construct the aggregation sequences (i.e. <code>hParamsAggGNN['nNodes']</code> is greater than one), then we might want to later bring together the features learned at every node to process them and obtain some final global feature (for instance, mapping into the number of classes). In that case, we need to further define the dimensions of a final aggregation MLP that acts on the concatenation of all the features learned by each separate node as input, and output the number of features specified in this list.

# #### Selection GNN (with zero-padding) <a class="anchor" id="subsubsec:SelGNNhyper"></a>

# When using the [selection GNN](#subsubsec:SelGNN) we operate directly on the graph signal $\mathbf{x}^{f}$, exchanging information through the $N$-node graph by means of the GSO $\mathbf{S}$. At each layer we have a graph convolution operation, a pointwise nonlinearity, and a graph pooling operation. Let's start with the convolution operation.
# 
# For any layer $\ell$, the input is some graph signal $\tilde{\mathbf{x}}_{\ell-1}^{g} \in \mathbb{R}^{N}$ that represents the $g$th feature, for $g=1,\ldots,F_{\ell-1}$. We then combine this $F_{\ell-1}$ features through a bank of $F_{\ell-1}F_{\ell}$ graph filters, giving the output $\tilde{\mathbf{u}}_{\ell}^{f} \in \mathbb{R}^{N}$ representing the $f$th output features, $f=1,\ldots,F_{\ell}$.
# 
# $$ \tilde{\mathbf{u}}_{\ell}^{f} = \sum_{g=0}^{F_{\ell-1}} \mathbf{h}_{\ell}^{fg} \ast_{\mathbf{S}} \tilde{\mathbf{x}}_{\ell-1}^{g} = \sum_{g=0}^{F_{\ell-1}} \left( \sum_{k=0}^{K_{\ell-1}} h_{\ell k}^{fg} \mathbf{S}^{k} \right) \tilde{\mathbf{x}}_{\ell-1}^{g} $$
# 
# What we learn through the selection GNN are the $K_{\ell}$ filter coefficients $\{h_{\ell k}^{fg} \}$ corresponding to the $F_{\ell}F_{\ell-1}$ filter banks that we have at layer $\ell$. So, for each layer, we need to specify: the number of input features $F_{\ell-1}$, the number of output features $F_{\ell}$, and the number of filter taps $K_{\ell}$. The input to the first layer is $\tilde{\mathbf{x}}_{0}^{f} = \mathbf{x}^{f}$ and, again, we have $F_{0}=F=1$ input features. In this case in particular, we consider a two-layer selection GNN, where in the first layer we output $F_{1} = 5$ features, and in the second output also $F_{2} =5$ features. The number of filter taps is $K_{1} = K_{2} = 3$ on each layer (information up to the $2$-hop neighborhood).

# In[27]:


hParamsSelGNN = {} # Create the dictionary to save the hyperparameters
hParamsSelGNN['name'] = 'SelGNN' # Name the architecture

hParamsSelGNN['F'] = [1, 5, 5] # Features per layer (first element is the number of input features)
hParamsSelGNN['K'] = [3, 3] # Number of filter taps per layer
hParamsSelGNN['bias'] = True # Decide whether to include a bias term


# Next, we apply a pointwise nonlinearity $\sigma_{\ell}$
# 
# $$ \mathbf{v}_{\ell}^{f} = \sigma_{\ell} \left( \mathbf{u}_{\ell}^{f} \right) $$
# 
# We adopt the same pointwise nonlinearity for all layers $\sigma_{\ell}=\sigma$, and since it is a pointwise nonlinearity, we use the ones defined in the PyTorch library <code>torch.nn</code>.

# In[28]:


hParamsSelGNN['sigma'] = nn.ReLU # Selected nonlinearity


# Finally, we move to the pooling operation. After the nonlinearity, we apply a summarizing function $\rho_{\ell}$ over the $\alpha_{\ell}$-hop neighborhood (typically, take the maximum of the signal at the nodes in the $\alpha_{\ell}$-hop neighborhood), followed by a downsampling, carried out by a sampling matrix $\mathbf{C}_{\ell}$, which is a binary $N_{\ell} \times N_{\ell-1}$ matrix, such that $\mathbf{C}_{\ell} \mathbf{1} = \mathbf{1}$ and $\mathbf{C}_{\ell}^{\mathsf{T}} \mathbf{1} \leq \mathbf{1}$. Then,
# 
# $$ \mathbf{x}_{\ell}^{f} = \mathbf{C}_{\ell} \rho_{\ell} \left( \mathbf{v}_{\ell}^{f} ; \mathbf{S}, \alpha_{\ell}\right) $$
# 
# The hyperparameters to determine, then, are the summarizing function $\rho = \rho_{\ell}$ (we adopt a max function, the same for all layers), the size of the neighborhoods $\alpha_{\ell}$ that we summarize at each layer, and the number of nodes that we keep at each layer $N_{\ell}$ (in the implementation of this code, we always take $\mathbf{C}_{\ell}$ to be the matrix that selects the first $N_{\ell}$, so we only need to determine the number of nodes $N_{\ell}$ that we need to keep; note that the nodes should be ordered following some importance criteria, where the more important nodes are located at the top of the vector -in this example we choose the degree-criteria, situating the largest-degree nodes at the top of the vector, although other criteria is available-).
# 
# For the summarizing function, we use the <code>MaxPoolLocal</code> class (availabe in <code>Utils.graphML</code>). We cannot use the regular pooling provided in <code>torch.nn</code> since we need to take into account the neighborhood (i.e. we cannot just take the maximum of contiguous elements). The size of the neighborhood to summarize is $\alpha_{1}=2$ and $\alpha_{2}=3$. The number of nodes to keep is $N_{1}=10$ at the output of the first layer and $N_{2}=5$ at the output of the second layer.

# In[29]:


hParamsSelGNN['rho'] = gml.MaxPoolLocal # Summarizing function
hParamsSelGNN['alpha'] = [2, 3] # alpha-hop neighborhood that
hParamsSelGNN['N'] = [10, 5] # Number of nodes to keep at the end of each layer is affected by the summary
hParamsSelGNN['D'] = [graphTools.build_duplication_matrix(nNodes)] #List of duplication matrices used for computing the GSO by using a parameter alpha 
hParamsSelGNN['D'] += [graphTools.build_duplication_matrix(N) for N in hParamsSelGNN['N']]
# We have to specify the criteria by which the nodes are selected (i.e. which 10 nodes are selected after the first layer). We also follow the degree criteria (i.e. the 10 nodes with largest degree). Other criteria can be found in the corresponding explanation of Aggregation GNN.

# In[30]:


hParamsSelGNN['order'] = 'Degree'


# After defining the hyperparameters of the graph convolutional layers, we apply a final MLP layer to adapt the dimensions to that of the number of classes (if we want to apply a deeper MLP, we add more elements to the list, each element determining how many output features after each fully connected layer, and we note that the nonlinearity applied between the layers is the same determined before in <code>hParamsSelGNN['sigma']</code>; the last layer has no nonlinearity applied since the softmax is applied by the loss function).

# In[31]:


hParamsSelGNN['dimLayersMLP'] = [nClasses] # Dimension of the fully connected layers after the GCN layers


# Before closing this section, we [recall](#subsubsec:SelGNN) the connection between the different quantities involved in the graph convolutional layer of a Selection GNN that we defined in the introduction.
# 
# The convolution is carried out over the original graph, as described by the GSO $\mathbf{S} \in \mathbb{R}^{N \times N}$
# 
# $$ \tilde{\mathbf{u}}_{\ell}^{f} = \sum_{g=0}^{F_{\ell-1}} \mathbf{h}_{\ell}^{fg} \ast_{\mathbf{S}} \tilde{\mathbf{x}}_{\ell-1}^{g} = \sum_{g=0}^{F_{\ell-1}} \left( \sum_{k=0}^{K_{\ell-1}} h_{\ell k}^{fg} \mathbf{S}^{k} \right) \tilde{\mathbf{x}}_{\ell-1}^{g} $$
# 
# The input to the graph convolution is the tilde quantity $\tilde{\mathbf{x}}_{\ell-1}^{g} \in \mathbb{R}^{N}$ which is the zero-padded graph signal defined over all the $N$ nodes, $\tilde{\mathbf{x}}_{\ell-1}^{g} = \mathbf{D}_{\ell-1}^{\mathsf{T}} \mathbf{x}_{\ell-1}^{g}$, for $\mathbf{D}_{\ell} = \mathbf{C}_{\ell} \mathbf{C}_{\ell-1} \cdots \mathbf{C}_{1} \in \{0,1\}^{N_{\ell} \times N}$. That is, we take the output of the $\ell-1$ layer, $\mathbf{x}_{\ell-1}^{g}$ for $g=1,\ldots,F_{\ell-1}$, which is defined only over $N_{\ell-1} \leq N$ nodes, and we zero pad it to fit the graph, so that the interactions with the graph shift operator $\mathbf{S}$ can be carried out. Here, each $\mathbf{C}_{\ell} \in \{0,1\}^{N_{\ell} \times N_{\ell-1}}$ selects $N_{\ell}$ nodes out of the previously selected $N_{\ell-1}$ nodes, with $N_{\ell} \leq N_{\ell-1}$. Thus, the concatenation of all $\mathbf{C}_{\ell}$ to build $\mathbf{D}_{\ell} \in \{0,1\}^{N_{\ell} \times N}$, becomes the selection matrix that takes the $N_{\ell}$ nodes directly out of the original $N$ nodes. As such, $\mathbf{D}_{\ell}^{\mathsf{T}}$ maps the selected $N_{\ell}$ nodes back to the $N$ nodes in the graph, zero-padding all the nodes that were not selected.
# 
# The convolution is then computed over the original graph $\mathbf{S} \in \mathbb{R}^{N \times N}$, operating on the zero-padded signal $\tilde{\mathbf{x}}_{\ell-1}^{g} \in \mathbb{R}^{N}$, and yielding another graph signal $\tilde{\mathbf{u}}_{\ell}^{f} \in \mathbb{R}^{N}$ at the output (this quantity has a tilde, since it is also defined over the original $N$-node graph). However, we only care about the value of this signal at the selected $N_{\ell}$ nodes. As such, we need to downsample to keep only these values: $\mathbf{u}_{\ell}^{f} = \mathbf{D}_{\ell} \tilde{\mathbf{u}}_{\ell}^{f}$.
# 
# In analogy with convolutional layers in the regular domain, we can view the graph convolutional layer as taking in the output of the previous layer in the appropriate dimensions (after pooling) $\mathbf{x}_{\ell-1}^{g} \in \mathbb{R}^{N_{\ell-1}}$, for $g=1,\ldots,F_{\ell-1}$, and yielding an output of the same dimensions $\mathbf{u}_{\ell}^{f} \in \mathbb{R}^{N_{\ell-1}}$, for $f=1,\ldots,F_{\ell}$ (where the convolutional layer just updated the number of features, from $F_{\ell-1}$ to $F_{\ell}$ but still yields data in the same dimension $N_{\ell}$ since we have not done yet any pooling for the layer $\ell$). From this viewpoint, the graph convolutional layer becomes
# 
# $$ \mathbf{u}_{\ell}^{f} = \mathbf{D}_{\ell-1} \sum_{g=0}^{F_{\ell-1}} \left( \sum_{k=0}^{K_{\ell-1}} h_{\ell k}^{fg} \mathbf{S}^{k} \right) \mathbf{D}_{\ell-1}^{\mathsf{T}} \mathbf{x}_{\ell-1}^{g} = \sum_{g=0}^{F_{\ell-1}} \left( \sum_{k=0}^{K_{\ell-1}} h_{\ell k}^{fg} \mathbf{S}_{\ell}^{(k)} \right) \mathbf{x}_{\ell-1}^{g} $$
# 
# where $\mathbf{S}_{\ell}^{(k)} = \mathbf{D}_{\ell-1}\mathbf{S}^{k}\mathbf{D}_{\ell-1}^{\mathsf{T}} \in \mathbf{R}^{N_{\ell-1} \times N_{\ell-1}}$ is a lower-dimensional matrix.
# 
# In essence, the graph convolutional layer implemented in <code>Utils.graphML.GraphFilter</code> takes as input the data $\mathbf{x}_{\ell-1}^{g} \in \mathbb{R}^{N_{\ell-1}}$ for $g=1,\ldots,F_{\ell-1}$ and gives as output the data $\mathbf{u}_{\ell}^{f} \in \mathbb{R}^{N_{\ell-1}}$ for $f=1,\ldots,F_{\ell}$, in analogy with the corresponding <code>torch.nn.Conv1d</code>. The corresponding zero-padding is handled internally. Then, the output of this layer can be fed directly into the pointwise nonlinearity $\sigma$ and the pooling function $\rho$ with downsampling $\mathbf{C}_{\ell}$ yielding $\mathbf{x}_{\ell}^{f} \in \mathbb{R}^{N_{\ell}}$ for $f=1,\ldots,F_{\ell}$ as described in the previous cell.

# Do not forget to save the corresponding hyperparameters in the <code>.txt</code> file, and add this architecture to the list.

# In[32]:


writeVarValues(varsFile, hParamsSelGNN)
modelList += [hParamsSelGNN['name']]


# #### Selection GNN (with graph coarsening) <a class="anchor" id="subsubsec:CrsGNNhyper"></a>

# The graph convolutional layer takes as input $\mathbf{x}_{\ell-1}^{g} \in \mathbb{R}^{N_{\ell-1}}$ for $g=1,\ldots,F_{\ell-1}$ and outputs $\mathbf{u}_{\ell}^{f} \in \mathbb{R}^{N_{\ell-1}}$ for $f=1,\ldots,F_{\ell}$, carrying out the operation
# 
# $$ \mathbf{u}_{\ell}^{f} = \sum_{g=0}^{F_{\ell-1}} \left( \sum_{k=0}^{K_{\ell-1}} h_{\ell k}^{fg} \mathbf{S}_{\ell}^{(k)} \right) \mathbf{x}_{\ell-1}^{g} $$
# 
# with $\mathbf{S}_{\ell}^{(k)} \in \mathbb{R}^{N_{\ell-1} \times N_{\ell-1}}$. While in the zero-padding case we use $\mathbf{S}_{\ell}^{(k)} =  \mathbf{D}_{\ell-1}\mathbf{S}^{k}\mathbf{D}_{\ell-1}^{\mathsf{T}}$, in the graph coarsening case, we use $\mathbf{S}_{\ell}^{(k)} = \mathbf{S}_{\ell}^{k}$ for some GSO $\mathbf{S}_{\ell}$ of smaller dimension, corresponding to a coarsened graph. In other words, we determine a set of graphs with GSOs $\{\mathbf{S}_{1},\ldots,\mathbf{S}_{L}\}$ where each successive GSO $\mathbf{S}_{\ell} \in \mathbb{R}^{N_{\ell-1} \times N_{\ell-1}}$ has a decreasing number of nodes $N_{\ell} \leq N_{\ell-1}$. These graphs are obtained by means of graph coarsening strategies. We set $\mathbf{S}_{1} = \mathbf{S}$ to be the original GSO.
# 
# For more details on the graph coarsening strategy, refer to <a href="https://openreview.net/forum?id=DQNsQf-UsoDBa">here</a> and <a href="https://papers.nips.cc/paper/6081-convolutional-neural-networks-on-graphs-with-fast-localized-spectral-filtering.pdf">here</a>. The original implementation of the code we take for the graph coarsening strategy can be obtained <a href="https://github.com/mdeff/cnn_graph">here</a>. The original implementation is in TensorFlow and we tried to copy it verbatim (with due credit to the original authors), except for those adaptations required to run it in PyTorch. Details are in the code.
# 
# The pointwise nonlinearities $\sigma$ and the pooling functions $\rho$ remain the same. As such, we just copy the same hyperparameters as the selection GNN with zero-padding. We change the name, and use the regular pooling operation provided in <code>torch.nn</code> because the graph coarsening algorithm already orders the nodes expecting them to be pooled together in contiguous fashion. We save the hyperparameters, and add it to the list.

# In[33]:


hParamsCrsGNN = deepcopy(hParamsSelGNN)
hParamsCrsGNN['name'] = 'CrsGNN'
hParamsCrsGNN['rho'] = nn.MaxPool1d
hParamsCrsGNN['order'] = None # We don't need any special ordering, since
    # it will be determined by the hierarchical clustering algorithm

writeVarValues(varsFile, hParamsCrsGNN)
modelList += [hParamsCrsGNN['name']]


# ### Logging parameters <a class="anchor" id="subsec:loggingParameters"></a>
# Finally, we handle the logging parameters (screen printing options, figure printing options, etc.)

# In[34]:


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


# ## Model Initialization <a class="anchor" id="sec:modelInitialization"></a>
# Now that we have created the dataset, and we have already defined all the hyperparameters for the architectures, and the loss function, and the optimizer that we are going to use, we can go ahead and initialize the corresponding architectures and bind them together with the loss function and the optimizer into the model.
# 
# The three architectures considered in this tutorial example ([Aggregation GNN](#subsubsec:AggGNN) and the two variants of [Selection GNN](#subsubsec:SelGNN) -zero-padding and graph coarsening-) are already created in the library <code>Modules.architectures</code>, so we just need to initialize them.
# 
# The <code>Model</code> class that is available in the <code>Modules.model</code> binds together the architecture, the loss function and the optimizer, as well as a name for the architecture and a specific directory where to save the model parameters. It also provides useful methods such as saving and loading architecture and optimizer parameters.

# We create a dictionary to save the initialized models, associated to their name.

# In[42]:


modelsGNN = {}


# ### Aggregation GNN <a class="anchor" id="subsec:AggGNNmodel"></a>

# In[43]:


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


# Do not forget to initialize the loss function before binding it within the <code>model</code> class. Recall that if more than one node is selected by setting <code>hParamsAggGNN['nNodes']</code> greater than one, then the output of the architecture will be another graph signal with the number of features learned by each node. If we want to further consolidate this features into a single, centralized feature, we need to further define another MLP <code>hParamsAggGNN['dimLayersAggMLP']</code> that acts on the concatenation of the features learned by all nodes to learn a single set of features (typically, the number of classes). This is invoked by key argument <code>dimLayersAggMLP = </code>, which otherwise is set to an empty list <code>[]</code> by default. If the number of nodes selected is 1, then the output is not a graph signal but just the collection of features collected at that node (and can be readily used, for instance, for classification).

# ### Selection GNN (with zero-padding) <a class="anchor" id="subsec:SelGNNmodel"></a>

# In[44]:


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
                     customLoss(nn.CrossEntropyLoss, multipliers),
                     thisOptim,
                     trainer,
                     evaluator,
                     device,
                     thisName,
                     saveDir)

#\\\ Add model to the dictionary
modelsGNN[thisName] = SelGNN


# ### Selection GNN (with graph coarsening) <a class="anchor" id="subsec:CrsGNNmodel"></a>

# In[45]:


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


# ## Training <a class="anchor" id="sec:training"></a>
# Now, we have created the graph and the corresponding datasets, we have created all the models that we want to train. So all that is left, is to train them.
# 
# Each model is binded together in the <code>Model</code> class which has a built-in <code>.train</code> method that trains each model specifically. This method outputs the loss and cost functions on the training and validation sets as a function of the training steps. So we will save them and plot them later on.

# In[46]:


lossTrain = {}
costTrain = {}
lossValid = {}
costValid = {}


# And now we can indeed train the models

# In[47]:

assert False

for thisModel in modelsGNN.keys():
    if thisModel == 'SelGNN':
        trainingOptions['learnGraph'] = True
    else:
        trainingOptions['learnGraph'] = False
        
    print("Training model %s..." % thisModel, end = ' ', flush = True)
    
    #Train
    thisTrainVars = modelsGNN[thisModel].train(data, nEpochs, batchSize, **trainingOptions)
    # Save the variables
    lossTrain[thisModel] = thisTrainVars['lossTrain']
    costTrain[thisModel] = thisTrainVars['costTrain']
    lossValid[thisModel] = thisTrainVars['lossValid']
    costValid[thisModel] = thisTrainVars['costValid']
    
    print("OK", flush = True)


# ## Evaluation <a class="anchor" id="sec:evaluation"></a>
# Once the models are trained, we evaluate their performance on the test set. We evaluate the performance of all the models, both for the <code>Best</code> model parameters (that is the parameters obtained for the lowest validation cost) and for the <code>Last</code> model parameters (the parameters obtained at the end of the training).
# 
# First, we create dictionaries to record the best and last cost results associated to each model. In the problem of source localization, the cost is the error rate.

# In[48]:


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
        lossTrain[thisModel] = lossTrain[thisModel][selectSamplesTrain]
        costTrain[thisModel] = costTrain[thisModel][selectSamplesTrain]
# And same for the validation, if necessary.
if xAxisMultiplierValid > 1:
    selectSamplesValid = np.arange(0, len(lossValid[thisModel]),                                    xAxisMultiplierValid)
    for thisModel in modelList:
        lossValid[thisModel] = lossValid[thisModel][selectSamplesValid]
        costValid[thisModel] = costValid[thisModel][selectSamplesValid]


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
