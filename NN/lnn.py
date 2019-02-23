#-*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def init_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)
    for l in range(1,L):
        parameters["W"+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
        parameters["b"+str(l)] = np.zeros((layer_dims[l],1))
    return parameters

def linear_forward(A,W,b):
    Z = np.dot(W,A) + b
    cache = (A,W,b)
    return Z,cache

def linear_activation_forward(A_pre,W,b,activation):
    #print A_pre
    #print W
    #print b
    if activation == 'sigmoid':
        Z,linear_cache = linear_forward(A_pre,W,b)
        A,activation_cache = sigmoid(Z)
    elif activation == 'relu':
        Z,linear_cache = linear_forward(A_pre,W,b)
        A,activation_cache = relu(Z)
    assert(A.shape ==(W.shape[0],A_pre.shape[1]))

    cache = (linear_cache,activation_cache)
    return A,cache

def forward(X,parameters):
    caches = []
    L = len(parameters) / 2
    A = X
    for l in range(1,L):
        A_prev = A
        A,cache = linear_activation_forward(A_prev,parameters["W"+str(l)],parameters["b"+str(l)],'sigmoid')
        caches.append(cache)
        #print parameters["W"+str(l)].shape
        #print "[A]"+str(A.shape)
    AL,cache = linear_activation_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],'sigmoid')
    caches.append(cache)
    assert(AL.shape == (10,X.shape[1]))

    #print parameters["W" + str(L)].shape
    #print "[A]" + str(AL.shape)
    return AL,caches

def linear_backward(dZ,cache):
    A_prev,W,b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ,A_prev.T)/m
    db = np.sum(dZ,axis=1,keepdims=True)/m
    dA_prev = np.dot(W.T,dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

def linear_activation_backward(dA,cache, activation):
    linear_cache,activation_cache = cache

    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev,dW,db = linear_backward(dZ,linear_cache)
    elif activation =='relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev,dW,db

def backward(AL,Y,caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) #???????

    current_cache = caches[L-1]
    grads["dA"+str(L)],grads["dW"+str(L)],grads["db"+str(L)] = linear_activation_backward(dAL,current_cache, activation="sigmoid")
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+2)],current_cache,activation="sigmoid")
        grads["dA" + str(l+1)] = dA_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp

    return grads


def cost_function(AL,Y,parameters):
    WSum = 0
    m = Y.shape[1]
    L = len(parameters)/2
    #cost = np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))/(-m)
    cost = np.sum(np.sqrt((AL - Y)**2))/m
    #for l in range(L):
    #    WSum += np.sum(parameters["W"+str(l+1)]**2)
    #WSum = WSum /(2.0*m)
    #cost += WSum

    cost = np.squeeze(cost)
    assert(cost.shape==())
    return cost

def sigmoid(z):
    result = 1.0/(1.0+np.exp(-z))
    cache = z
    return result,z

def relu(z):
    result = np.maximum(z,0)
    cache = z
    return result,z

def sigmoid_backward(dA,activation_cache):
    Z = activation_cache
    g,_ = sigmoid(Z)
    gD = g * (1.0 - g)
    dZ = dA * gD
    return dZ

def relu_backward(dA,activation_cache):
    Z = activation_cache
    Z[Z<0] = 0
    Z[Z>=0] = 1
    dZ = dA * Z
    return dZ

def update_paremeters(parameters,grads,learning_rate):
    L = len(parameters)/2
    for l in range(L):
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learning_rate * grads["dW"+str(l+1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters

def oneHot(Y,numLabel):
	m = Y.shape[1]
	y_v = np.empty((numLabel,m))
	for i in range(m):
		y_v[int(Y[0,i]),i] = 1	

	return y_v
