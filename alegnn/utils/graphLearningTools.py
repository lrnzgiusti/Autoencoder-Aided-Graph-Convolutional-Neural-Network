#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 15:19:05 2021

@author: ince
"""
import torch
import numpy as np
torch.set_default_dtype(torch.float64)

def build_elimination_matrix(N : int, sparse : bool = False) -> torch.Tensor:
    """
    
    Build a sparse elimination matrix

    Parameters
    ----------
    N : int
        The dimension of the adjacency matrix.

    sparse : bool
        Wether or not the result is sparse
    Returns
    -------
    torch.sparse.FloatTensor
        sparse representation of elimination matrix.

    """
    T = np.triu(np.ones(N))
    f = np.where(T.flatten())[0]
    k = N*(N+1)//2
    nsq = N**2
    x = f + nsq*np.arange(0,k)
    idxs = np.unravel_index(x, (k, nsq))
    vals = torch.ones((len(idxs[0]),), dtype=torch.float32)
    idxs = np.row_stack(idxs)  #np.column_stack((idxs[0],idxs[1]))
    E = torch.sparse.FloatTensor(torch.from_numpy(idxs), 
                                    vals, 
                                    (k, nsq)).coalesce().double()
    if sparse:
        return E
    return E.to_dense()

def build_duplication_matrix(N : int, sparse : bool  = False) -> torch.Tensor:
    """
    Build a sparse duplication matrix

    Parameters
    ----------
    N : int
        The dimension of the adjacency matrix.
    sparse : bool
        Wether or not the result is sparse

    Returns
    -------
    torch.sparse.FloatTensor
        sparse representation of duplication matrix.

    """
    m   = N * (N + 1) // 2;
    nsq = N**2;
    r   = 1;
    a   = 1;
    v   = np.zeros(nsq+1, dtype=np.int32);
    for i in range(1, N+1):
       b = i;
       for j in range(0, i-2+1):
          v[r] = b;
          b    = b + N - j - 1;
          r    = r + 1;
       
       for j in range(0, N-i+1):
         v[r] = a + j;
         r    = r + 1;
       
       a = a + N - i + 1;
    v = v[1:]-1
    rows = np.arange(0,nsq, dtype=np.int32)
    vals = torch.ones((len(v)), dtype=torch.float32)
    idxs = [[rows[i], v[i]] for i in range(nsq)]
    idxs = torch.Tensor(idxs).type(torch.LongTensor).t()
    D = torch.sparse.FloatTensor(idxs, 
                                    vals, 
                                    (nsq, m)).coalesce().double()
    if sparse:
        return D
    return D.to_dense()



def get_diagonal_and_off_diag_idxs_fast(N: int) -> tuple:
    """
      
      Get the indices of the elements of a vech representation of an adjacency
      that will fall onto the main diagonal and onto the upper tiangular
      
      TODO: implementing this with a tensor representation will speed up?
      
      Parameters
      ----------
      N : int
      The number of nodes in the graph domain.
      
      Returns
      -------
      indeces : list
      indices of the main diagonal.
      
      indeces_not_diag : numpy array
      indices of the upper triangular matrix.
      
    """
    k = N
    i = 0
    indeces = []
    indeces_not_diag = []
    while i < (N*(N+1)//2) :
        indeces.append(i)
        if len(indeces_not_diag) == 0:
            indeces_not_diag = list(range(i+1,i+k))
        else:
            indeces_not_diag.extend(list(range(i+1,i+k)))
    
        i += k
        k -= 1
    return torch.tensor(indeces, dtype=torch.long), \
           torch.tensor(indeces_not_diag, dtype=torch.long)

def projection_onto_constraint(alpha, reciproc_indices, N):
    r"""
    
    Project a vector onto the feasible set, see: arxiv:1309.1541
    Parameters
    ----------
    alpha : torch.Tensor
        vector that describes the main diagonal and upper triangular part 
        of the adjacency matrix taken on the values
        that fall off the main diagonal.
        
    reciproc_indices : torch.Tensor
        1/j for j in [1, N*(N+1)//2-N+1)) useful for large values of N
                      
    N : int
        The dimension of the adjacency matrix.

    Returns
    -------
    alpha_pi : torch.Tensor
        the vector alpha, taken on the values off the main diagonal
        projected onto the feasible set .

    """
    alpha_sorted, _  = torch.sort(alpha, dim=0, descending=True)
    cum_sum_alpha    = (N/2.) - torch.cumsum(alpha_sorted, dim=0) 
    cum_mean         = torch.mul(reciproc_indices, cum_sum_alpha)
    alpha_tilde      = torch.clamp(alpha_sorted + cum_mean, min=0) #clip
    where            = torch.nonzero(alpha_tilde)
    multiplier       = cum_mean[where[-1][0]]
    return           torch.relu(alpha+multiplier)

def alpha_step(alpha, constants):
    """
    
    Puts the values onto the main diagonal to zero
    Project the values onto the upper triangular part to the feasible set
    
    THE ORDER OF THESE OPERATIONS CHANGES THE RESULT

    Parameters
    ----------
    alpha : torch.Tensor
        vector that describes the main diagonal and upper triangular part 
        of the adjacency matrix taken entirely.
        
    N : int
        The dimension of the adjacency matrix.
        
    reciproc_indices : torch.Tensor
        1/j for j in [1, N*(N+1)//2-N+1)) useful for large values of N
                      
    idx_diag : torch.Tensor // list
      indices of the main diagonal.
      
    idx_not_diag : torch.Tensor // numpy array 
      indices of the upper triangular part of the adjacency matrix.
      

    Returns
    -------
    This function acts inplace.

    """
    
    
    #project the values of alpha that are off the diagonal
    #onto the feasible set
    #alpha_ = alpha.detach()
    alpha[constants.idx_diag] = torch.zeros((len(constants.idx_diag), 1))
    alpha[constants.idx_not_diag] = projection_onto_constraint(alpha[constants.idx_not_diag], 
                                                        constants.reciproc_indices, 
                                                        constants.N )
    #return alpha_
   

    


class Constants:
    """
    Constants: useful constants that flows across the entire learning procedure
    
    """
    def __init__(self, N):
        self.N = N
        
        self.E, self.D,   \
        self.idx_diag, self.idx_not_diag, \
        self.reciproc_indices = self.compute_one_store_forever(N)

    def compute_one_store_forever(self, N):
        duplication = build_duplication_matrix(N, sparse=True)
        elimination = build_elimination_matrix(N, sparse=True)
        idx_diag, idx_not_diag = get_diagonal_and_off_diag_idxs_fast(N)
        reciproc_indices = torch.Tensor([1/j for j in range(1,N*(N+1)//2-N+1)])\
                                .reshape((N*(N+1)//2-N,1))
        return elimination, \
               duplication, \
               idx_diag,    \
               idx_not_diag, \
               reciproc_indices
    
   