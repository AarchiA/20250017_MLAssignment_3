#Q6

import numpy as np
import matplotlib.pyplot as plt
import math

#Defining Activation functions
def relu(x):
    return np.maximum(0,x)

def relu_grad(Z):
    g = np.zeros(Z.shape)
    g[Z>0] =1 
    return g

def softmax(Z):
    sm = np.exp(Z)/(np.sum(np.exp(Z),axis = 0))
    return sm

def identity(x):
    return x

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return (1.0/(1.0+ np.exp(-x)))

def norm_softmax(Z):
    b = Z.max(axis=0)
    y = np.exp(Z-b)
    A = y/(np.sum(y,axis = 0))
    return A
# Used for initializing parameter for the given layer size   
def parameters_init(l_size):
    parameters = {}                                                                 
    for i in range(len(l_size) - 1):
        parameters['Weight'+str(i+1)] = np.random.randn(l_size[i+1],l_size[i])
        parameters['Bias'+str(i+1)] = np.zeros((l_size[i+1],1))        
    return parameters

# Used for performing the forward pass
def forward_propagation(X, parameters, dataset='Digits'):
    v= {}                                                                     
    d = len(parameters) // 2
    A_i = X
    v['Acti'+str(0)] = X
    #For layers not including outer layer
    for l in range(1,d):                                                      
        A_i_1 = A_i
        Z = np.dot(parameters['Weight'+str(l)],A_i_1) + parameters['Bias'+str(l)]
        v['Z'+str(l)] = Z 
        A_i = relu(Z)
        v['Acti'+str(l)] = relu(Z)
    #For outer layer
    Z = np.dot(parameters['Weight'+str(d)],A_i) + parameters['Bias'+str(d)]                     
    v['Z'+str(d)] = Z 
    
    if dataset =='Digits':
        A_n = norm_softmax(Z)
    else:
        A_n = Z
    
    return A_n,v

# Used in calculating gradients
def grad_c(dZ,gradients,parameters,v,l,m):           
    dW = (1/m)* np.dot(dZ,v['Acti'+str(l-1)].T)
    gradients['Weight'+str(l)] = dW
    db = (1/m)* np.sum(dZ)
    gradients['Bias'+str(l)] = db
    dA_i_1 = np.dot(parameters['Weight'+str(l)].T,dZ)
    gradients['Acti'+str(l-1)] = dA_i_1
    return gradients

# Used for back propagation
def back_propagation(A_n,Y,v,parameters,dataset='Digits'):                                       
    gradients = {}
    L = len(parameters) //2
    m = A_n.shape[1]
    
    #For first layer backpropagation
    dZ = A_n-Y
    gradients = grad_c(dZ,gradients,parameters,v,L,m)                   
    
    #Remaining layers back propagation
    for l in reversed(range(1,L)):                                              
        dA = gradients['Acti'+str(l)]
        dZ = dA * relu_grad(v['Z'+str(l)])     
        gradients = grad_c(dZ,gradients,parameters,v,l,m)               
    
    return gradients

# Used for updating parameters     
def u_parameters(parameters, gradients, lr):
    l_size = len(parameters)//2
    new_parameters = {}

    for i in range(1,l_size+1):
        new_parameters.update({"Weight" + str(i) : (parameters["Weight" + str(i)] - lr * gradients["Weight" + str(i)])})
        new_parameters.update({'Bias' + str(i) : (parameters['Bias' + str(i)] - lr * gradients['Bias' + str(i)])})
    
    return new_parameters

# Used for classification, we do one_hot encoding for the target value
def represent_one_hot(y):                                                                 
    C = np.unique(y).size
    y_hot = np.eye(C)[:,y.reshape(-1)]
    
    return y_hot

# Used to predict class in classification
def multiClass_predict(X,parameters):
    A_n,_ = forward_propagation(X,parameters)
    pred = np.argmax(A_n,axis=0)
    return pred

# The model function
def model(X,Y,layer_sizes,lr=0.01,num_epochs=1000,dataset='Digits'):        
                                                                                
    parameters = parameters_init(layer_sizes)
    for i in range(num_epochs):
        if dataset=='Digits':
              A_n,v = forward_propagation(X,parameters, dataset='Digits')
              gradients = back_propagation(A_n,Y,v,parameters, dataset='Digits')
              parameters = u_parameters(parameters,gradients,lr)
        else:
            for i in range(num_epochs):
                A_n,v = forward_propagation(X, parameters, dataset='Boston')
                gradients = back_propagation(A_n, Y, v, parameters, dataset='Boston')
                parameters = u_parameters(parameters, gradients, lr)
    return parameters

#Digits Dataset
#Importing library and dataset
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Loading dataset
X,Y = load_digits(return_X_y=True) 
#Initial accuracy 
initial_acc = 0.0

#Doing cross validation, calling the model and finding accuracy
for k in range(3):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
    X_train, X_test = X_train.T, X_test.T
    # For making  one hot representation 
    one_hot_Y_train = represent_one_hot(Y_train)                                              
    Xn_train = X_train/255
    Xn_test = X_test/255
    layer_dims = [X_train.shape[0],10,one_hot_Y_train.shape[0]]
    #Calling model
    parameters = model(Xn_train,one_hot_Y_train,layer_dims,lr=0.5,num_epochs=1000, dataset='Digits')
    np.savez('finalParameters',parameters=parameters)
    Weights = np.load('finalParameters.npz', allow_pickle=True)
    finalParameters = Weights['parameters'].item()
    #Predicting
    pred_y = multiClass_predict(Xn_test,finalParameters)                         
    accuracy = accuracy_score(Y_test, pred_y)
    #Finding maximum Accuracy
    if accuracy > initial_acc:
        initial_acc = accuracy

print("Using 3-fold cross validation the max accuracy for Digits Dataset ", initial_acc*100)


#Digits Dataset
#Importing library and dataset
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Loading dataset
X,Y = load_digits(return_X_y=True) 
#Initial accuracy 
initial_acc = 0.0

#Doing cross validation, calling the model and finding accuracy
for k in range(3):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
    X_train, X_test = X_train.T, X_test.T
    # For making  one hot representation 
    one_hot_Y_train = represent_one_hot(Y_train)                                              
    Xn_train = X_train/255
    Xn_test = X_test/255
    layer_dims = [X_train.shape[0],10,one_hot_Y_train.shape[0]]
    #Calling model
    parameters = model(Xn_train,one_hot_Y_train,layer_dims,lr=0.5,num_epochs=1000, dataset='Digits')
    np.savez('finalParameters',parameters=parameters)
    Weights = np.load('finalParameters.npz', allow_pickle=True)
    finalParameters = Weights['parameters'].item()
    #Predicting
    pred_y = multiClass_predict(Xn_test,finalParameters)                         
    accuracy = accuracy_score(Y_test, pred_y)
    #Finding maximum Accuracy
    if accuracy > initial_acc:
        initial_acc = accuracy

print("Using 3-fold cross validation the max accuracy for Digits Dataset ", initial_acc*100)


#Output
#Using 3-fold cross validation the max accuracy for Digits Dataset  90.74074074074075

#Q6
#Boston Dataset
#Importing libraries and dataset 
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Loading dataset and initialization
data = load_boston()                                                              
X,Y = data["data"], data["target"]  
layer_dims = [13, 10, 1]
initial_rmse = float('inf')

#For calculating RMSE
def RMSE(X_test,Y_test, parameters):
    x,y = forward_propagation(X_test, params, dataset = 'Boston')
    test_acc = mean_squared_error(Y_test, x.T)
    return test_acc

#Doing cross validation, calling function and finding RMSE
for k in range(3):                                                            
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3)
    X_train, X_test = X_train.T, X_test.T
    params = model(X_train,Y_train,layer_dims,lr=0.01,num_epochs=100, dataset='Boston')
    rmse = RMSE(X_test, Y_test, params)
    if rmse < initial_rmse:
        initial_rmse = rmse 

print("Using 3-fold cross validation RMSE for Boston Dataset ",str(initial_rmse))


#Output
#Using 3-fold cross validation RMSE for Boston Dataset  60.96255735815568

