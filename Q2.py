#Q2 
#L1 and L2 regularised logistic regression for 2-class problem

import math
import pandas as pd
import autograd.numpy as np
from autograd import grad

#For importing dataset	
from sklearn.datasets import load_breast_cancer     
from sklearn.model_selection import train_test_split
X,Y = load_breast_cancer(return_X_y=True)

class LR() :
  #Initialization
  def __init__( self, alpha, n, reg, reg_value=0.5) :		
    self.alpha = alpha		
    self.n = n
    self.reg = reg
    self.reg_value = reg_value
  
  #Doing the L1 Regularization
  def l1_reg(self, W):     
    A = 0.5*(np.tanh(np.dot(self.X, self.W)/2) + 1)
    prob = np.dot(A, self.Y) + np.dot((1 - A), (1 - self.Y))
    return -np.sum(np.log(prob)) + self.reg_value * np.sum(self.W)

  #Doing the L2 Regularization
  def l2_reg(self, W):     
    A = 0.5*(np.tanh(np.dot(self.X, self.W)/2) + 1)
    prob = np.dot(A, self.Y) + np.dot((1 - A), (1 - self.Y))
    return -np.sum(np.log(prob)) + self.reg_value * np.dot(self.W, np.transpose(self.W))

  #Updating the weights 
  def up_w( self ) :		
    A = 0.5*(np.tanh(np.dot(self.X, self.W)/2) + 1)
    t = np.reshape( A - self.Y.T, self.m )		

    #Autograd For doing L1 Regularization
    if self.reg=='l1reg':
        dW = grad(self.l1_reg)(self.W)
    #Autograd For doing L2 Regularization
    elif self.reg=='l2reg':
        dW = grad(self.l2_reg)(self.W)
    #Weight update
    db = np.sum( t ) / self.m
    self.W = self.W - self.alpha * dW	    
    self.b = self.b - self.alpha * db
    return self

  #Fit Function
  def fit( self, X, Y ) :		
    self.m, self.n = X.shape			
    self.W = np.zeros( self.n )				
    self.X = X		
    self.Y = Y
    self.b = 0
    for i in range( self.n ) :			
      self.up_w()			
    return self
	#Predict function
  def predict( self, X ) :	
    Z = 1 / ( 1 + np.exp( - ( X.dot( self.W)) +self.b ) )
    Y = np.where( Z >= 0.5, 1, 0 )	
    return Y, Z

import warnings
warnings.filterwarnings( "ignore" )

reg_values = [0.01,0.1,0.2,0.3,0.4,0.5]
accuracy1 = 0
accuracy2 = 0
parameter = 0

#Finding optimal value of lambda parameter with the help of nested cross validation
for i in range(len(reg_values)):  
    for k in range(3):        
        X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/3, random_state = None, shuffle=True)     #Splitting the dataset
        model = LR( alpha = 0.01, n = 5000, reg='l1reg', reg_value=reg_values[i] )                                         #Calling the LR function
        model = model.fit( X_train, Y_train )	                                                                             # Training the model 
        Y_pred, Y_prob = model.predict( X_test )                                                                           #Doing the prediction for test dataset 
        acc = np.mean(Y_pred == Y_test.flatten()) * 100                                                                    #Calculating Accuracy 
        if acc > accuracy1:
            accuracy1 = acc
    if accuracy1 > accuracy2:
            parameter = i
            accuracy2 = accuracy1
print( "->The optimal value lambda parameter : ", reg_values[parameter])
print( "->By using Autograd and regularization technique we get maximum Accuracy  as", accuracy2)

#Output
#->The optimal value lambda parameter :  0.3
#->By using Autograd and regularization technique we get maximum Accuracy as 45.78947368421053
