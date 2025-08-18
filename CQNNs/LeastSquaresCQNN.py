"""
Author: Zachary Yetman Van Egmond
Code related to the implementaiton of least squares trained QNNs.

References:
- L. Rodrigues, “Least squares solution for training of two-layer quadratic neural networks with applications to system identification,” 
  in International Conference on Control, Decision and Information Technologies (CoDIT), Vallette, Malta, July 2024, pp. 916–921
"""

import numpy as np


def vecf(v, f):
    """
    Converts a colummn vector v into vector of all quadratic combinations within filter size f
    (e.g. for v=[v1 v2 v3 v4 v5] and f=3, vecf(v,f) will contain v1v2, v1v2, v1v3 but not v1v4 or v1v5)
    Adapted from Luis Rodrigues' MATLAB code
    """
    if len(v.shape) == 1:
        v = np.expand_dims(v, axis=1)
    if v.shape[1] > v.shape[0]:
        v = v.T

    n, _ = v.shape
    if f > n:
        raise Exception(f"Filter length {f} too large for data with {n} features")
    vv = v @ v.T
    out = np.empty((1,int(0.5*f*(2*n-f+1))))
    start_index = 0
    for i in range(f):
        out[0,start_index:start_index+n-i] = np.diag(vv,i)
        start_index += n-i
    return out


def CQNN_LS_H(x, f, a=0.0937, b=0.5, c=0.4688):
    """
    Returns the CQNN least-square equivilant H matrix (y=Hw) from singlechannel data x
    """
    dp, n = x.shape
    n_vec = int(0.5*f*(2*n - f + 1))
    H = np.zeros(shape=(dp,n_vec+n))
    H1 = np.zeros(shape=(dp,n_vec))
    H2 = np.zeros(shape=(dp,n_vec))
    X = np.zeros(shape=(dp,n))
    for i in range(dp):
        v = a*vecf(x[i,:], f)
        H1[i,:] = v
        X[i,:] = np.expand_dims(x[i,:], axis=0)
    H2[:,:n] = c*np.ones(shape=(dp,n))
    H2[:,n:] = np.zeros(shape=(dp,n_vec-n))
    H[:,:n_vec] = H1+H2
    H[:,n_vec:] = b*X

    return H


def CQNN_Multichannel_LS_H(x, f, a=0.0937, b=0.5, c=0.4688):
    """
    Returns the CQNN least-square equivilant H matrix (y=Hw) from multichannel data x
    Stacks the Hc matrices of each data channel to form the overall H matrix
    """
    dp, n_channels, n = x.shape
    n_vec = int(0.5*f*(2*n - f + 1))
    H = np.zeros(shape=(dp,n_channels*(n_vec+n)))
    for channel in range(n_channels):
        Hc = np.zeros(shape=(dp,n_vec+n))
        H1 = np.zeros(shape=(dp,n_vec))
        H2 = np.zeros(shape=(dp,n_vec))
        X = np.zeros(shape=(dp,n))
        for i in range(dp):
            v = a*vecf(x[i,channel,:], f)
            H1[i,:] = v
            X[i,:] = np.expand_dims(x[i,channel,:], axis=0)
        H2[:,:n] = c*np.ones(shape=(dp,n))
        H2[:,n:] = np.zeros(shape=(dp,n_vec-n))
        Hc[:,:n_vec] = H1+H2
        Hc[:,n_vec:] = b*X
        H[:,channel*(n_vec+n):(channel+1)*(n_vec+n)] = Hc

    return H


class LSCQNN:
    def __init__(self):
        self.weights = None
        self.a = None
        self.b = None
        self.c = None
        self.f = None
    
    def Train(self, x, y, f, beta=0, a=0.0937, b=0.5, c=0.4688):
        H_ls = CQNN_LS_H(x, f, a, b, c)
        w_ls = np.linalg.inv(H_ls.T @ H_ls + beta*np.eye(H_ls.shape[1])) @ H_ls.T @ y
        self.weights = w_ls
        self.a = a
        self.b = b
        self.c = c
        self.f = f
    
    def Eval(self, x):
        H = CQNN_LS_H(x, self.f, self.a, self.b, self.c)
        y = H @ self.weights
        return y
