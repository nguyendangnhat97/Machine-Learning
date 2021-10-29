# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 10:22:34 2021

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
iris = datasets.load_iris() #load iris dataset (flower data)
iris_X = iris.data
iris_y = iris.target
X0 = iris_X[iris_y == 0,:]
X1 = iris_X[iris_y == 1,:]
X2 = iris_X[iris_y == 2,:]
print(iris_X)
print(iris_y)


X_train, X_test, y_train, y_test = train_test_split(
    iris_X, iris_y, test_size=50)

print('Training size : %d' %len(y_train))
print('Test size: %d' %len(X_test))

clf = neighbors.KNeighborsClassifier(n_neighbors=10, p = 2, weights='distance')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Print result for 20 test data points:')
print("Predicted labels: ", y_pred[20:40])
print('Ground truth:     ', y_test[20:40])


#evaluation
from sklearn.metrics import accuracy_score
print('Accuracy of 1 NN: %.2f %%' %(100*accuracy_score(y_test, y_pred)))
