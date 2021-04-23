#Q1
#Unregularised Logistic regression for 2-class problem

import math
import pandas as pd
import autograd.numpy as np
from autograd import grad

# Dataset	
from sklearn.datasets import load_breast_cancer     
from sklearn.model_selection import train_test_split

class Un_LR() :
  #Initialzation 
  def __init__( self, alpha, n_iterations, grad = "GradDesc" ) :		
    self.alpha = alpha		
    self.n_iterations = n_iterations
    self.grad = grad

  #Calculating J_theta  
  def J_Theta(self, W):     
    A = 0.5*(np.tanh(np.dot(self.X, self.W)/2) + 1)
    label_probabilities = np.dot(A, self.Y) + np.dot((1 - A), (1 - self.Y))
    return -np.sum(np.log(label_probabilities))
  
  #Updating the weights     
  def up_w( self ) :		
    A = 0.5*(np.tanh(np.dot(self.X, self.W)/2) + 1)
    t = ( A - self.Y.T )		
    t = np.reshape( t, self.m )		
    if self.grad == "GradDesc":
        dW = np.dot( self.X.T, t ) / self.m		
        db = np.sum( t ) / self.m
    elif self.grad == "Autograd":
        dW = grad(self.J_Theta)(self.W)
        db = np.sum( t ) / self.m
    self.W = self.W - self.alpha * dW	    	
    self.b = self.b - self.alpha * db
    return self

  #Fit function for learning
  def fit( self, X, Y ) :		
    self.m, self.n = X.shape			
    self.W = np.zeros( self.n )				
    self.X = X		
    self.Y = Y
    self.b = 0
				
    for i in range( self.n_iterations ) :			
      self.up_w()			
    return self

  #Predict function 
  def predict( self, X, partial = False ) :	
    if partial:
        Z = 1 / ( 1 + np.exp( - ( X.dot( self.W[:2] ) +self.b ) ) )
        Y = np.where( Z >= 0.5, 1, 0 )	
    else:
        Z = 1 / ( 1 + np.exp( - ( X.dot( self.W)) +self.b ) )
        Y = np.where( Z >= 0.5, 1, 0 )	
    return Y, Z

import warnings
warnings.filterwarnings( "ignore" )

#Here used K-Fold Cross Validation for GradDesc(a)
X,Y = load_breast_cancer(return_X_y=True)
max_acc=0
for k in range(3):     
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/3, random_state = None, shuffle=True)    #Splitting the data  
    model = Un_LR( alpha = 0.01, n_iterations = 1000, grad = "GradDesc" )     #Calling the Un_LR function 
    model.fit( X_train, Y_train )	    #Training	the model
    
    Y_pred, Y_prob = model.predict( X_test )    #Doing the prediction for test dataset 
    acc = np.mean(Y_pred == Y_test.flatten()) * 100 #Calculating accuracy
    if acc > max_acc: #Finding max accuracy
        max_acc = acc

print( "->Logistic Regression model using GradDesc gives max accuracy of : ",max_acc)

#Here used K-Fold Cross Validation for Auto(b)
X,Y = load_breast_cancer(return_X_y=True)
max=0
for k in range(3):    
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/3, random_state = None, shuffle=True)     #Splitting the data   
    model = Un_LR( alpha = 0.01, n_iterations = 10000, grad = "Autograd" )   #Calling the Un_LR function   
    model = model.fit( X_train, Y_train )	 #Training	the model   
    Y_pred, Y_prob = model.predict( X_test )    #Doing the prediction for test dataset  
    acc = np.mean(Y_pred == Y_test.flatten()) * 100 #Calculating accuracy
    if acc > max: #Finding max accuracy
        max = acc

print( "->Logistic Regression model using Autograd gives max accuracy of : ", max)

# Plotting Decision Boundary
import matplotlib.pyplot as plt

X_plot = X_train[:,:2]        

colors = ['g','b']    
for idx,cl in enumerate(np.unique(Y_train)):
    plt.scatter(x = X_train[:,0][np.where(Y_train == cl)[0]], y = X_train[:,1][np.where(Y_train == cl)[0]], alpha = 0.8, c = colors[idx], marker = 'o', label = cl, s = 30)
            
model = Un_LR( alpha = 0.01, n_iterations = 1000, grad = "GradDesc" )     
model.fit( X_plot, Y_train )	    	
Y_plot = []

for i in X_plot:
  Y_plot.append(-i[0]*model.W[0]/model.W[1] + model.b)

Y_plot = np.array(Y_plot)
x_min, x_max = np.argmin(X_plot[:,0]), np.argmax(X_plot[:,0])
X_plot = X_plot[[x_min,x_max],1]
Y_plot = Y_plot[[x_min,x_max]]
plt.plot(X_plot, Y_plot)

plt.legend(loc = 'best')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

Y_pred, Y_prob = model.predict( X_test[:,:2] )    
acc = np.mean(Y_pred == Y_test.flatten()) * 100
print('-> Overall Accuracy is', acc)

#Output 
#->Logistic Regression model using GradDesc gives max accuracy of :  87.89473684210526
#->Logistic Regression model using Autograd gives max accuracy of :  38.421052631578945
#-> Overall Accuracy is 50.0
