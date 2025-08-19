"""
Author: Zachary Yetman Van Egmond
Code related to the implementaiton of least squares trained CQNNs.

References:
- Z. Yetman Van Egmond and L. Rodrigues, “Least squares training of quadratic convolutional neural networks 
  with applications to system theory,” 2024, arXiv:2411.08267.
"""

import numpy as np


def vecf(v, f):
    """
    Converts a colummn vector v into vector of all quadratic combinations within filter size f
    (e.g. for v=[v1 v2 v3 v4 v5] and f=3, vecf(v,f) will contain v1v2, v1v2, v1v3 but not v1v4 or v1v5)
    Adapted from Luis Rodrigues' MATLAB code

    :param v: Column vector
    :param int f: Filter size
    :return out: Row vector of quadratic combinations
    """

    # Reshape input v to expected shape
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
    Returns the CQNN least-square equivilant H matrix (y=Hw) from singlechannel data x (Yetman Van Egmond, 2024)

    :param nparray x: Nxn array of inputs (N: Datapoints, n: features)
    :param int f: Filter size
    :param float a,b,c: Quadratic parameters
    :return H: Regressor matrix
    """

    dp, n = x.shape
    n_vec = int(0.5*f*(2*n - f + 1))

    H = np.empty((dp,n_vec+n))
    H1 = np.empty((dp,n_vec))
    H2 = np.zeros((dp,n_vec))
    X = np.empty((dp,n))
    for i in range(dp):
        v = a*vecf(x[i,:], f)
        H1[i,:] = v
        X[i,:] = np.expand_dims(x[i,:], axis=0)
    H2[:,:n] = c*np.ones((dp,n))
    H[:,:n_vec] = H1+H2
    H[:,n_vec:] = b*X

    return H


def CQNN_Multichannel_LS_H(x, f, a=0.0937, b=0.5, c=0.4688):
    """
    Returns the CQNN least-square equivilant H matrix (y=Hw) from multichannel data x (Yetman Van Egmond, 2024)
    Stacks the Hc matrices of each data channel to form the overall H matrix

    :param nparray x: NxCxn array of inputs (N: Datapoints, C: channels, n: features per channel)
    :param int f: Filter size
    :param float a,b,c: Quadratic parameters
    :return H: Regressor matrix
    """
    dp, n_channels, n = x.shape
    n_vec = int(0.5*f*(2*n - f + 1))

    H = np.empty((dp,n_channels*(n_vec+n)))
    for channel in range(n_channels):
        Hc = np.empty((dp,n_vec+n))
        H1 = np.empty((dp,n_vec))
        H2 = np.zeros((dp,n_vec))
        X = np.empty((dp,n))
        for i in range(dp):
            v = a*vecf(x[i,channel,:], f)
            H1[i,:] = v
            X[i,:] = np.expand_dims(x[i,channel,:], axis=0)
        H2[:,:n] = c*np.ones((dp,n))
        Hc[:,:n_vec] = H1+H2
        Hc[:,n_vec:] = b*X
        H[:,channel*(n_vec+n):(channel+1)*(n_vec+n)] = Hc

    return H


def Z_LSCQNN_Weights(weight_vector, f, a=None, b=None, c=None):
    """
    Constructs the Z matrix from the LS-CQNN weight vector. 
    Z matrix does not contain quadratic parameters if a, b, or c is set to None.

    :param weight_vector: Weight vector from a LSQNN
    :param float f: Filter size
    :param float a,b,c: Quadratic parameters
    :return Z: Z matrix of QNN qudaratic form
    """

    r, _ = weight_vector.shape
    n = int((f**2 - f + 2*r)/(2*(f+1)))

    Z1 = np.zeros(shape=(n,n))
    Z2 = np.zeros(shape=(n,1))
    Z4 = np.zeros(shape=(1,1))

    num_weights = weight_vector.shape[0]
    num_Z1_weights = num_weights - n

    diag_count = 0
    diag_entry = 0
    for i in range(num_weights):
        if i < num_Z1_weights:
            if diag_count == 0:
                Z1[diag_entry,diag_entry] = weight_vector[i][0]
                Z4[0,0] = Z4[0,0] + weight_vector[i][0]
                diag_entry += 1
                if diag_entry == n:
                    diag_count += 1
                    diag_entry = 0
            else:
                Z1[diag_entry+diag_count,diag_entry] = weight_vector[i][0]
                Z1[diag_entry,diag_entry+diag_count] = weight_vector[i][0]
                diag_entry += 1
                if diag_entry == n - diag_count:
                    diag_count += 1
                    diag_entry = 0
        else:
            Z2[i-num_Z1_weights,0] = weight_vector[i][0]
    
    if (a != None) and (b != None) and (c != None):
        Z = np.bmat([[a*Z1, 0.5*b*Z2], [0.5*b*Z2.T, c*Z4]])
    else:
        Z = np.bmat([[Z1, Z2], [Z2.T, Z4]])

    return Z


class LSCQNN:
    """Least squares trained convolution quadratic neural network"""
    def __init__(self):
        self.weights = None
        self.a = None
        self.b = None
        self.c = None
        self.f = None
        self.multichannel = None
    
    def Train(self, x, y, f, beta=0, a=0.0937, b=0.5, c=0.4688, multi_channel=False):
        """
        Train the CQNN

        :param nparray x: Nxn (singlechannel) or NxCxn (multichannel) array of inputs (N: Datapoints, C: channels, n: features per channel)
        :param nparray y: Nxp array of labels (N: Datapoints, p: outputs)
        :param int f: Filter size
        :param float beta: Regularization coefficient
        :param float a,b,c: Quadratic parameters
        :param bool multi_channel: If true, uses the multichannel formulation for CQNNs
        """
        
        #Calculate regressor matrix
        if multi_channel:
            H_ls = CQNN_Multichannel_LS_H(x, f, a, b, c)
        else:
            H_ls = CQNN_LS_H(x, f, a, b, c)
        # Calulcated weights
        w_ls = np.linalg.inv(H_ls.T @ H_ls + beta*np.eye(H_ls.shape[1])) @ H_ls.T @ y

        self.weights = w_ls
        self.a = a
        self.b = b
        self.c = c
        self.f = f
        self.multichannel = multi_channel
    
    def Eval(self, x):
        """
        Eval with the CQNN

        :param nparray x: Nxn (singlechannel) or NxCxn (multichannel) array of inputs (N: Datapoints, C: channels, n: features per channel)
        :return nparray y: Array of outputs
        """

        if self.multichannel:
            H = CQNN_Multichannel_LS_H(x, self.f, self.a, self.b, self.c)
        else:
            H = CQNN_LS_H(x, self.f, self.a, self.b, self.c)
        y = H @ self.weights
        return y
    
    def Z_matrix(self, quadratic_params=True):
        """
        Return the quaratic form of theC QNN weights

        :param bool quadratic_params: If true then the quadratic parameters a,b,c are pre integrated into Z)
        """
        if quadratic_params:
            return Z_LSCQNN_Weights(self.weights, f=self.f, a=self.a, b=self.b, c=self.c)
        return Z_LSCQNN_Weights(self.weights, f=self.f)
