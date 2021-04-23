#Q5
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
