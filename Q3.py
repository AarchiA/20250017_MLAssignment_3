#Q3 
#K class Logistic regression

import pandas as pd
import autograd.numpy as np
import matplotlib.pyplot as plt 
from autograd import grad
from scipy.optimize import fmin_cg

#Importing dataset
from sklearn.datasets import load_digits 

#For doing StratifiedKFold
from sklearn.model_selection import StratifiedKFold

def sigmoid(X,theta):
    return 0.5*(np.tanh(np.dot(X, theta.T)/2) + 1)

def RegCF(theta, X, y, lmbda): 
    l = len(y)
    t1 = np.multiply(np.log(sigmoid(X, theta)).T,y)
    t2 = np.multiply(np.log(1-sigmoid(X, theta)).T,(1-y))
    return np.sum(t1 + t2) /(-l) 

def RegG(theta, X, y, lmbda): 
    m = X.shape[0]
    y1 = sigmoid(X, theta)
    return (1/m) * np.dot(X.T, y1 - y)

import warnings
warnings.filterwarnings( "ignore" )

X, y = load_digits(n_class=10, return_X_y=True)
m = len(y)
ones = np.ones((m,1))
#for adding the value of intercept
X = np.hstack((ones, X)) 
(m,n) = X.shape
accuracy_list = []

#Doing the StratifiedKFolding(c)
s = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
plot_y = y  
for train_index, test_index in s.split(X, y):
    x_train_fold, x_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    value_lambda = 0.1
    classes = 10
    theta = np.zeros((classes,n)) 
    for i in range(classes):
        digit_class = i if i else 10
        theta[i] = fmin_cg(f = RegCF, x0 = theta[i],  fprime = grad(RegCF), args = (x_train_fold, (y_train_fold == digit_class).flatten(), value_lambda), maxiter = 10, disp =False)

    pred = np.argmax(x_test_fold @ theta.T, axis = 1)
    pred = [e if e else 10 for e in pred]
    a = np.mean(pred == y_test_fold.flatten()) * 100
    accuracy_list.append(a)
    plot_y = y_test_fold
print("-> Accuracy using Autograd is : ", np.mean(np.array(accuracy_list)))

accuracy_list = []

#Doing the StratifiedKFolding(c)
s = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
plot_y = y  
for train_index, test_index in s.split(X, y):
    x_train_fold, x_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    value_lambda = 0.1
    classes = 10
    theta = np.zeros((classes,n)) 
    for i in range(classes):
        digit_class = i if i else 10
        theta[i] = fmin_cg(f = RegCF, x0 = theta[i],  fprime = RegG, args = (x_train_fold, (y_train_fold == digit_class).flatten(), value_lambda), maxiter = 10, disp =False)

    pred = np.argmax(x_test_fold @ theta.T, axis = 1)
    pred = [e if e else 10 for e in pred]
    a = np.mean(pred == y_test_fold.flatten()) * 100
    accuracy_list.append(a)
    plot_y = y_test_fold
print("-> Accuracy without Autograd is : ", np.mean(np.array(accuracy_list)))


confusion_matrix = [[0 for i in range(10)] for i in range(10)]

for i in range(len(plot_y)):
  confusion_matrix[pred[i]%10][plot_y[i]%10] += 1 

print("-> The confusion matrix is :")
print(pd.DataFrame(confusion_matrix))

#Graph(d)
from sklearn.decomposition import PCA
x = PCA(2)  # project from 64 to 2 dimensions
projected = x.fit_transform(X)
plt.scatter(projected[:, 0], projected[:, 1], c=y, edgecolor='none', alpha=0.5, cmap='Spectral')
plt.xlabel('1st part')
plt.ylabel('2nd part')
plt.colorbar();

#Output
#-> Accuracy using Autograd is :  74.56904231625836
#-> Accuracy without Autograd is :  71.73372927493196
#-> The confusion matrix is :
#    0   1   2   3   4   5   6   7   8   9
#0   0   0   0   0   0   0   0   0   0   0
#1  43  42   1   7   1   3   3   3  43   5
#2   0   1  43   0   0   0   0   0   0   0
#3   0   0   0  38   0   0   0   0   0   0
#4   0   0   0   0  44   0   1   0   1   0
#5   1   0   0   0   0  41   0   0   0   1
#6   0   1   0   0   0   1  42   0   0   0
#7   0   0   0   0   0   0   0  41   0   0
#8   0   0   0   0   0   0   0   0   0   0
#9   0   2   0   1   0   0   0   0   0  39
