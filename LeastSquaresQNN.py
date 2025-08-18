"""
Author: Zachary Yetman Van Egmond
Code related to the implementaiton of least squares trained QNNs.

References:
- L. Rodrigues, “Least squares solution for training of two-layer quadratic neural networks with applications to system identification,” 
  in International Conference on Control, Decision and Information Technologies (CoDIT), Vallette, Malta, July 2024, pp. 916–921
"""

import numpy as np


def vec(v):
    """
    Converts a colummn vector v into vector of all quadratic combinations 
    Adapted from Luis Rodrigues' MATLAB code
    """
    if len(v.shape) == 1:
        v = np.expand_dims(v, axis=1)
    if v.shape[1] > v.shape[0]:
        v = v.T

    n, _ = v.shape
    vv = v @ v.T
    out = np.empty((1,int(0.5*n*(n+1))))
    start_index = 0
    for i in range(n):
        out[0,start_index:start_index+n-i] = np.diag(vv,i)
        start_index += n-i
    return out


def QNN_LS_H(x, a=0.0937, b=0.5, c=0.4688):
    """
    Returns the QNN least-square regressor matrix H from data x (Rodrigues, 2024)
    """
    dp, n = x.shape
    n_vec = int(n*(n+1)/2)
    H1 = np.zeros(shape=(dp,n_vec))
    X = np.zeros(shape=(dp,n))
    for i in range(dp):
        v = a*vec(x[i,:])
        H1[i,:] = v
        X[i,:] = np.expand_dims(x[i,:], axis=0)
    H2 = np.hstack([c*np.ones(shape=(dp,n)), np.zeros(shape=(dp,n_vec-n))])
    H = np.hstack([H1+H2, b*X])

    return H


def Z_LSQNN_Weights(weight_vector, a=None, b=None, c=None):
    """
    Constructs the Z matrix from the LS-QNN weight vector (theta). 
    Z matrix does not contain quadratic parameters if a,b, or c is set to None.
    """
    r, _ = weight_vector.shape
    n = int((np.sqrt(9 + 8*r) - 3)/2)
    Z_1_entries = int(n*(n+1)/2)

    Z = np.zeros(shape=(n+1,n+1))
    # Z_1 and Z_4
    diag = 0
    counter = 0
    for i in range(Z_1_entries):
        if i < n:
            Z[i,i] = weight_vector[i,0]
            Z[n,n] += weight_vector[i,0]
        else:
            Z[counter,counter+diag] = weight_vector[i,0]/2
            Z[counter+diag,counter] = weight_vector[i,0]/2
        counter += 1
        if counter == n - diag:
            counter = 0
            diag += 1
    
    # Z_2
    counter = 0
    for i in range(Z_1_entries, r):
        Z[n,counter] = weight_vector[i,0]
        Z[counter,n] = weight_vector[i,0]
        counter += 1
    
    if (a != None) and (b != None) and (c != None):
        Z[:-1,:-1] = a*Z[:-1,:-1]
        Z[-1,:-1] = 0.5*b*Z[-1,:-1]
        Z[:-1,-1] = 0.5*b*Z[:-1,-1]
        Z[-1,-1] = c*Z[-1,-1]
    
    return Z


class LSQNN:
    def __init__(self):
        self.weights = None
        self.a = None
        self.b = None
        self.c = None
    
    def Train(self, x, y, beta=0, a=0.0937, b=0.5, c=0.4688):
        H_ls = QNN_LS_H(x, a, b, c)
        w_ls = np.linalg.inv(H_ls.T @ H_ls + beta*np.eye(H_ls.shape[1])) @ H_ls.T @ y
        self.weights = w_ls
        self.a = a
        self.b = b
        self.c = c
    
    def Eval(self, x):
        H = QNN_LS_H(x, self.a, self.b, self.c)
        y = H @ self.weights
        return y
    
    def Z_matrix(self, quadratic_params=True):
        if quadratic_params:
            return Z_LSQNN_Weights(self.weights, a=self.a, b=self.b, c=self.c)
        return Z_LSQNN_Weights(self.weights)