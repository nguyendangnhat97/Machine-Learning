# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 13:40:21 2021

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('data_linear.csv').values
N = data.shape[0]
x = data[:, 0].reshape(-1, 1) #assign x values as area, then reshape the matrix
y = data[:, 1].reshape(-1, 1) #assign y values as price, then reshape the matrix
plt.scatter(x, y)
plt.xlabel('Area')
plt.ylabel('Price')

x = np.hstack((np.ones((N, 1)), x)) # for easier to multiply matrix
w = np.array([0.,1.]).reshape(-1, 1)

numOfIteration = 100
cost = np.zeros((numOfIteration, 1))
learning_rate = 0.000001
#gradient descent algorithm
for i in range (1, numOfIteration):
    r = np.dot(x, w) - y #r = f(x) - y
    cost[i] = np.sum(r*r) #loss function
    w[0] -= learning_rate*np.sum(r)
    w[1] -= learning_rate*np.sum(np.multiply(r, x[:, 1].reshape(-1, 1)))    
    print(cost[i])
#change numOfIteration until loss function is changed inconsiderably
predict = np.dot(x, w)
plt.plot((x[0][1], x[N-1][1]), (predict[0], predict[N-1]), 'red')
plt.show()
#predict example
x1 = 50
y1 = w[0] + w[1] * x1
print('Price for 50m^2 is : ', y1)