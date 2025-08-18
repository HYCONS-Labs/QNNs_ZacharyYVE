import numpy as np
import cvxpy as cp

def QNN(X_data, Y_data, norm=2, beta=0, a=0.0937, b=0.5, c=0.4688, verbose=True):
    """
    Solves the convex optimization dual problem for a 2-layer MLP with quadratic activation function from:
    B. Bartan and M. Pilanci, “Neural spectrahedra and semidefinite lifts: 
    Global convex optimization of polynomial activation neural networks in fully polynomial-time,” 2021.
    - Use "getResultsQNN" to process data with returned Z matrix

    Params:
    - X_data: NN input data, rows=samples, columns=features
    - Y_data: NN expected output, rows=samples, columns=output values
    - norm: (default:2) Loss function norm (2:norm-2, 1:norm-1, -1:norm-infinity)
    - beta: (default:0) Regularization coefficient
    - a, b, c: (default:0.0937, 0.5, 0.4688) Quadratic activation function coeffients f(z)=az^2+bz+c

    Returns:
    - Z: Quadratic matrix of shape (n+1 x n+1 x p) where n is feature length and p is output number
    - Zp, Zm: Matrices that make up Z
    """
    dp, n = X_data.shape
    _, p = Y_data.shape

    # Setup problem
    Zp = {}
    Zm = {}
    for k in range(p):
        Zp[k] = cp.Variable((n+1,n+1), symmetric=True)
        Zm[k] = cp.Variable((n+1,n+1), symmetric=True)

    # Construct Z matrix
    Z = {}
    for k in range(p):
        Zpk = Zp[k]
        Zmk = Zm[k]
        Z_tl = a*(Zpk[:n,:n]-Zmk[:n,:n])
        Z_tr = 0.5*b*(Zpk[:n,n:]-Zmk[:n,n:])
        Z_bl = 0.5*b*(Zpk[n:,:n]-Zmk[n:,:n])
        Z_br = c*(Zpk[n:,n:]-Zmk[n:,n:])
        Zk = cp.bmat([[Z_tl, Z_tr],
                      [Z_bl, Z_br]])
        Z[k] = Zk

    # Setup error vector to minimize
    y_error = []
    for i in range(dp):
        x_bar = np.vstack((np.expand_dims(X_data[i], axis=1), 1))
        y_k = []
        for k in range(p):
            Zk = Z[k]
            y_new_k = (x_bar.T @ Zk @ x_bar) - Y_data[i][k]
            y_k.append(y_new_k)
        y_error.append(y_k)
    y_error = cp.bmat(y_error)

    # Regularization
    reg_term = 0
    for k in range(p):
        Zpk = Zp[k]
        Zmk = Zm[k]
        reg_term = reg_term + (Zpk[n,n] + Zmk[n,n])
    reg_term = reg_term * beta

    # Solve problem
    if norm == -1:
        obj = cp.Minimize(cp.norm(y_error, "inf") + reg_term)
    elif norm == 1:
        obj = cp.Minimize(cp.norm(y_error, 1) + reg_term)
    elif norm == 2:
        obj = cp.Minimize(cp.norm(y_error, 2) + reg_term)
    else:
        raise Exception(f"Unrecongnized norm type '{norm}' (possible norms: -1(infinity), 1, 2)") 
    const = []
    for k in range(p):
        Zpk = Zp[k]
        Zmk = Zm[k]
        const.append(Zpk[n,n]==cp.trace(Zpk[0:n,0:n]))
        const.append(Zmk[n,n]==cp.trace(Zmk[0:n,0:n]))
        const.append(Zpk >> 0)
        const.append(Zmk >> 0)
    prob = cp.Problem(obj, const)
    result = prob.solve(solver=cp.MOSEK)
    # Print final values
    if verbose:
        print("---Optimization Results---")
        print("Status:", prob.status)
        # print("Zp.shape:", Zp_val.shape, "Zm.shape:", Zm_val.shape)
        print("----------")

    # Extract Z matrix
    Z = np.zeros(shape=(n+1,n+1,p))
    Zp_np = np.zeros(shape=(n+1,n+1,p))
    Zm_np = np.zeros(shape=(n+1,n+1,p))
    for k in range(p):
        Zpk = Zp[k].value
        Zmk = Zm[k].value
        Z_tl = a*(Zpk[:n,:n]-Zmk[:n,:n])
        Z_tr = 0.5*b*(Zpk[:n,n:]-Zmk[:n,n:])
        Z_bl = 0.5*b*(Zpk[n:,:n]-Zmk[n:,:n])
        Z_br = c*(Zpk[n:,n:]-Zmk[n:,n:])
        Zk = np.bmat([[Z_tl, Z_tr],
                    [Z_bl, Z_br]])
        Z[:,:,k] = Zk
        Zp_np[:,:,k] = Zpk
        Zm_np[:,:,k] = Zmk
    
    return Z, Zp_np, Zm_np


def CQNN(X_data, Y_data, f, norm=2, beta=0, a=0.0937, b=0.5, c=0.4688):
    """
    Solves the convex optimization dual problem for a 2-layer CNN with quadratic activation function from:
    B. Bartan and M. Pilanci, “Neural spectrahedra and semidefinite lifts: 
    Global convex optimization of polynomial activation neural networks in fully polynomial-time,” 2021.
    - Only supports single output and stride of 1
    - Use "getResultsCQNN" to process data with returned Z matrix

    Params:
    - X_data: CNN input data, rows=samples, columns=features
    - Y_data: CNN expected output, rows=samples, column=output value
    - f: Filter size
    - norm: (default:2) Loss function norm (2:norm-2, 1:norm-1, -1:norm-infinity)
    - beta: (default:0) Regularization coefficient
    - a, b, c: (default:0.0937, 0.5, 0.4688) Quadratic activation function coeffients f(z)=az^2+bz+c

    Returns:
    - Z: Quadratic matrix of shape (f+1 x f+1 x n-f+1)
    - Zp, Zm: Matrices that make up Z 
    """
    dp, n = X_data.shape
    _, p = Y_data.shape
    if p > 1:
        raise Exception("Unsupported Y array size")

    # Setup problem
    # Matrix sub-components
    Z1p = {}
    Z1m = {}
    Z2p = {}
    Z2m = {}
    Z4p = {}
    Z4m = {}
    # Complete matrix
    Z = {}
    Zp = {}
    Zm = {}
    for k in range(n - f + 1):
        # Sub components
        Z1p[k] = cp.Variable((f,f), symmetric=True)
        Z1m[k] = cp.Variable((f,f), symmetric=True)
        Z2p[k] = cp.Variable((f,1))
        Z2m[k] = cp.Variable((f,1))
        Z4p[k] = cp.Variable((1,1))
        Z4m[k] = cp.Variable((1,1))
        # Z matrix
        Z_tl = a*(Z1p[k]-Z1m[k])
        Z_tr = 0.5*b*(Z2p[k]-Z2m[k])
        Z_bl = (0.5*b*(Z2p[k]-Z2m[k])).T
        Z_br = c*(Z4p[k]-Z4m[k])
        Zk = cp.bmat([[Z_tl, Z_tr],
                      [Z_bl, Z_br]])
        Z[k] = Zk
        # Zp matrix
        Zpk = cp.bmat([[Z1p[k],   Z2p[k]],
                       [Z2p[k].T, Z4p[k]]])
        Zp[k] = Zpk
        # Zm matrix
        Zmk = cp.bmat([[Z1m[k],   Z2m[k]],
                       [Z2m[k].T, Z4m[k]]])
        Zm[k] = Zmk

    # Setup constraints
    const = []
    for k in range(n - f + 1):
        # Trace constraints
        const.append(Z4p[k]==cp.trace(Z1p[k]))
        const.append(Z4m[k]==cp.trace(Z1m[k]))
        # PSD constraints
        const.append(Zp[k] >> 0)
        const.append(Zm[k] >> 0)

    # Setup error vector to minimize
    y_error = []
    for i in range(dp):
        # Construct predicted y from sum of patches
        y_i = 0
        for k in range(n - f + 1):
            x_sample = X_data[i,k:k+f]
            x_bar = np.vstack((np.expand_dims(x_sample, axis=1), 1))
            Zk = Z[k]
            y_k = (x_bar.T @ Zk @ x_bar)
            y_i += y_k
        # Error = predicted y - actual y
        y_error.append(y_i - Y_data[i][0])
    y_error = cp.bmat(y_error)

    # Regularization
    reg_term = 0
    for k in range(n - f + 1):
        reg_term = reg_term + (Z4p[k] + Z4m[k])
    reg_term = reg_term * beta

    # Solve problem
    if norm == -1:
        obj = cp.Minimize(cp.norm(y_error, "inf") + reg_term)
    elif norm == 1:
        obj = cp.Minimize(cp.norm(y_error, 1) + reg_term)
    elif norm == 2:
        obj = cp.Minimize(cp.norm(y_error, 2) + reg_term)
    else:
        raise Exception(f"Unrecongnized norm type '{norm}' (possible norms: -1(infinity), 1, 2)") 
    prob = cp.Problem(obj, const)
    result = prob.solve(solver=cp.MOSEK)
    # Print final values
    print("---Optimization Results---")
    print("Status:", prob.status)
    
    # Extract Z matrix
    Z = np.zeros(shape=(f+1,f+1,n-f+1))
    Zp = np.zeros(shape=(f+1,f+1,n-f+1))
    Zm = np.zeros(shape=(f+1,f+1,n-f+1))
    for k in range(n-f+1):
        Z_tl = a*(Z1p[k].value-Z1m[k].value)
        Z_tr = 0.5*b*(Z2p[k].value-Z2m[k].value)
        Z_bl = (0.5*b*(Z2p[k].value-Z2m[k].value)).T
        Z_br = c*(Z4p[k].value-Z4m[k].value)
        Zk = np.block([[Z_tl, Z_tr],
                       [Z_bl, Z_br]])
        Zpk = np.block([[Z1p[k].value, Z2p[k].value],
                        [(Z2p[k].value).T, Z4p[k].value]])
        Zmk = np.block([[Z1m[k].value, Z2m[k].value],
                        [(Z2m[k].value).T, Z4m[k].value]])
        Z[:,:,k] = Zk
        Zp[:,:,k] = Zpk
        Zm[:,:,k] = Zmk
    print(f"Final Z shape: {Z.shape}")
    print("----------")
    
    return Z, Zp, Zm


def CQNN_QNNForm(X_data, Y_data, f, norm=2, a=0.0937, b=0.5, c=0.4688):
    """
    Solves the QNN form of the CQNN convex problem
    - Use "getResultsQNN" to process data with returned Z matrix
    - No regularization

    Params:
    - X_data: CNN input data, rows=samples, columns=features
    - Y_data: CNN expected output, rows=samples, columns=output values
    - f: Filter size
    - norm: (default:2) Loss function norm (2:norm-2, 1:norm-1, -1:norm-infinity)
    - a, b, c: (default:0.0937, 0.5, 0.4688) Quadratic activation function coeffients f(z)=az^2+bz+c

    Returns:
    - Z: Quadratic matrix of shape (n+1 x n+1 x p) where n is feature length, p is output number
    """
    dp, n = X_data.shape
    _, p = Y_data.shape

    if not (a > 0):
        raise Exception("Invalid value for 'a'")
    if not (c > 0):
        raise Exception("Invalid value for 'c'")
    if not ((b*b - 4*a*c) > 0):
        raise Exception("Invalid values for 'a', 'b', and 'c'")

    # Setup problem
    # Matrix sub-components
    Z1 = {}
    Z2 = {}
    Z4 = {}
    # Complete matrix
    Z = {}
    for k in range(p):
        # Sub components
        Z1[k] = cp.Variable((n,n), symmetric=True)
        Z2[k] = cp.Variable((n,1))
        Z4[k] = cp.Variable((1,1))
        # Z matrix
        Z_tl = a*Z1[k]
        Z_tr = 0.5*b*Z2[k]
        Z_bl = (0.5*b*Z2[k]).T
        Z_br = c*Z4[k]
        Zk = cp.bmat([[Z_tl, Z_tr],
                      [Z_bl, Z_br]])
        Z[k] = Zk

    # Setup constraints
    const = []
    for k in range(p):
        # Trace constraints
        const.append(Z4[k]==cp.trace(Z1[k]))
        # Zero constraints
        Z1k = Z1[k]
        for i in range(n):
            for j in range(n):
                if abs(i-j) >= f:
                    const.append(Z1k[i,j] == 0)
                    const.append(Z1k[j,i] == 0)

    # Setup error vector to minimize
    y_error = []
    for i in range(dp):
        x_bar = np.vstack((np.expand_dims(X_data[i], axis=1), 1))
        y_k = []
        for k in range(p):
            Zk = Z[k]
            y_new_k = (x_bar.T @ Zk @ x_bar) - Y_data[i][k]
            y_k.append(y_new_k)
        y_error.append(y_k)
    y_error = cp.bmat(y_error)

    # Solve problem
    if norm == -1:
        obj = cp.Minimize(cp.norm(y_error, "inf"))
    elif norm == 1:
        obj = cp.Minimize(cp.norm(y_error, 1))
    elif norm == 2:
        obj = cp.Minimize(cp.norm(y_error, 2))
    else:
        raise Exception(f"Unrecongnized norm type '{norm}' (possible norms: -1(infinity), 1, 2)") 
    prob = cp.Problem(obj, const)
    result = prob.solve(solver=cp.MOSEK)
    # Print final values
    print("---Optimization Results---")
    print("Status:", prob.status)
    # print("Zp.shape:", Zp_val.shape, "Zm.shape:", Zm_val.shape)
    print("----------")

    # Extract Z matrix
    Z = np.zeros(shape=(n+1,n+1,p))
    for k in range(p):
        Z_tl = a*Z1[k].value
        Z_tr = 0.5*b*Z2[k].value
        Z_bl = (0.5*b*Z2[k].value).T
        Z_br = c*Z4[k].value
        Zk = np.bmat([[Z_tl, Z_tr],
                    [Z_bl, Z_br]])
        Z[:,:,k] = Zk
    
    return Z


def getResultsQNN(X, Z):
    """
    Returns the predicted outputs for given input data and QNN Z matrix

    Params:
    - X: Input data
    - Z: QNN Z matrix

    Returns:
    - Y_pred: Predicted output values
    """

    dp, n = X.shape
    _, _, p = Z.shape

    Y_pred = np.zeros(shape=(dp,p))

    # Get output for each input sample
    for i in range(dp):
        x_bar = np.vstack((np.expand_dims(X[i], axis=1), 1))
        y_k = []
        for k in range(p):
            Zk = Z[:,:,k]
            y_new_k = (x_bar.T @ Zk @ x_bar).item()
            y_k.append(y_new_k)
        Y_pred[i,:] = y_k
    
    return Y_pred


def getResultsCQNN(X, Z):
    """
    Returns the predicted outputs for given input data and CQNN Z matrix

    Params:
    - X: Input data
    - Z: CQNN Z matrix

    Returns:
    - Y_pred: Predicted output values
    """
    dp, n = X.shape
    fp1, _, _ = Z.shape
    f = fp1 - 1

    Y_pred = np.zeros(shape=(dp,1))
    
    # Get output for each input sample
    for i in range(dp):
        y_i = 0
        for k in range(n - f + 1):
            x_sample = X[i,k:k+f]
            x_bar = np.vstack((np.expand_dims(x_sample, axis=1), 1))
            Zk = Z[:,:,k]
            y_k = (x_bar.T @ Zk @ x_bar)
            y_i += y_k
        Y_pred[i,0] = y_i[0][0]

    return Y_pred


def Zcqnn_to_Zqnn(Z_cqnn):
    """
    Converts a CQNN Z matrix to an eqvuiliant QNN Z matrix (can be used with "getResultsQNN")

    Params:
    - Z_cqnn: CQNN Z matrix (returned by "CQNN" function)

    Returns:
    - Z_qnn: QNN Z matrix equivilant (like those returned by "QNN" function)
    """
    fp1, _, nmfp1 = Z_cqnn.shape
    f = fp1 - 1
    n = nmfp1 + f - 1

    # Construct QNN Z matrix
    Z1 = np.zeros(shape=(n,n))
    Z2 = np.zeros(shape=(n,1))
    Z4 = np.zeros(shape=(1,1))
    for k in range(n - f + 1):
        Z1[k:k+f,k:k+f] = Z1[k:k+f,k:k+f] + Z_cqnn[:f,:f,k]
        Z2[k:k+f,0] = Z2[k:k+f,0] + Z_cqnn[:f,f,k]
        Z4[0,0] = Z4[0,0] + Z_cqnn[f,f,k]
    Z_qnn = np.bmat([[Z1, Z2],
                     [Z2.T, Z4]])
    
    # Add a dimension to make it a proper 1 output matrix
    Z_qnn = np.expand_dims(Z_qnn, axis=2)

    return Z_qnn