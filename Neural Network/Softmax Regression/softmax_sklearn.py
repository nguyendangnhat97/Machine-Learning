# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 15:38:39 2021

@author: Admin
"""
import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from scipy import sparse
np.random.seed(20)
means = [[2,2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis = 0) #stack X1,X2,X3
#extended data

C = 3
label = np.asarray([0]*N + [1]*N + [2]*N).T
W_init = np.random.randn(X.shape[0], C)

logreg = linear_model.LogisticRegression(C=3, solver = 'lbfgs', multi_class='multinomial')
logreg.fit(X, label)
print('W:', logreg.coef_)
print('bias:', logreg.intercept_)
