# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 14:47:01 2021

@author: Admin
"""
import numpy as np
from scipy import sparse
def softmax(Z):
    e_Z = np.exp(Z - np.max(Z))
    A = e_Z/e_Z.sum(axis = 0)
    return A

N = 2
d = 2
C = 3

X = np.random.randn(d, N)
y = np.random.randint(0, 3, (N,))

#one-hot coding
def convert_label(y, C = C):
    '''
    convert 1d label to a matrix label: each column of this 
    matrix coresponding to 1 element in y. In i-th column of Y, 
    only one non-zeros element located in the y[i]-th position, 
    and = 1 ex: y = [0, 2, 1, 0], and 3 classes then return

            [[1, 0, 0, 1],
             [0, 0, 1, 0],
             [0, 1, 0, 0]]
    '''
    Y = sparse.coo_matrix((np.ones_like(y), 
        (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y

Y = convert_label(y, C)

#cost or loss function
def cost(X, Y, W):
    A = softmax(W.T.dot(X))
    return -np.sum(Y*np.log(A))

W_init = np.random.randn(d, C)
#gradient
def grad(X, Y, W):
    A = softmax(W.T.dot(X))
    E = A - Y
    return X.dot(E.T)

#gradient checking function
def numerical_grad(X, Y, W, cost):
    eps = 1e-6
    g = np.zeros_like(W)
    for i in range (W.shape[0]):
        for j in range(W.shape[1]):
            W_p = W.copy()
            W_n = W.copy()
            W_p[i, j] += eps
            W_n[i, j] -= eps
            g[i,j] = (cost(X, Y, W_p) - cost(X, Y, W_n))/(2*eps)
    return g
g1 = grad(X, Y, W_init)
g2 = numerical_grad(X, Y, W_init, cost)
print('norm g1 and g2', np.linalg.norm(g1-g2))

#Main function
def softmax_regression(X, y, W_init, eta, tol = 1e-4, max_count = 10000):
    W = [W_init] # +1 dimension
    C = W_init.shape[1]
    Y = convert_label(y, C)
    it = 0
    N = X.shape[1]
    d = X.shape[0]
    
    count = 0
    check_w_after = 20
    
    while count < max_count:
        #mix data
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = Y[:, i].reshape(C, 1)
            ai = softmax(np.dot(W[-1].T, xi))
            W_new = W[-1] + eta * xi.dot((yi-ai).T)
            count += 1
            #stopping criteria
            if count%check_w_after == 0:
                if np.linalg.norm(W_new - W[-check_w_after]) < tol:
                    return W
            W.append(W_new)
        print('count', count)
        return W
eta = .05
d = X.shape[0]
W_init = np.random.randn(d, C)
W = softmax_regression(X, y, W_init, eta)


#predict function after W is found
def pred(W, X):
    A = softmax(W[-1].T.dot(X))
    return np.argmax(A, axis = 0) #indices of cluster
print('w', W)
print('pred', pred(W, X))