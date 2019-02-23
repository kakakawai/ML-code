# -*- coding:utf-8 -*-

import lnn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import scipy.io as sio

def LoadData():
    print "[+]Loading Data"
    data = sio.loadmat("ex4data1.mat")
    weights = sio.loadmat('ex4weights.mat')
    #print np.shape(data['X'])
    #print np.zeros(np.shape(weights['Theta1']))
    #print np.shape(data['y'])
    #print data['y']
    #print np.shape(weights['Theta1'])
    #print np.shape(weights['Theta2'])
    #print "[+]Loading Data ....OK"

    return [data['X'].T, data['y'].T, weights['Theta1'], weights['Theta2']]


def predict(X,Y,parameters):
    result,_ = lnn.forward(X,parameters)
    result = result.argmax(axis=0)
    result += 1
    m = len(result)
    Accuracy = 1 - (np.count_nonzero(y - result) / (1.0 * m))
    print '[+]Accuracy:%f' % (Accuracy)
    return Accuracy


costList = []
aList = []

layer_dims = [400,25,10]
learning_rate = 0.5
decay_rate = 0.997

X, y, Theta1, Theta2 = LoadData()
m = X.shape[1]
y_v = np.zeros((layer_dims[-1],m))
parameters = lnn.init_parameters(layer_dims)
for i in range(m):
    y_v[int(y[0,i])-1,i] = 1
t1 = time.time()
for i in range(3000):
    AL,caches = lnn.forward(X,parameters)
    #print caches
    cost = lnn.cost_function(AL,y_v,parameters)
    #print cost
    costList.append(cost)
    aList.append(learning_rate)
    grads = lnn.backward(AL,y_v,caches)
    parameters = lnn.update_paremeters(parameters,grads,learning_rate)
    if not i%10:
        learning_rate = learning_rate * decay_rate
    print "\r[+]%d/3000\r" % (i)
t2 = time.time()
predict(X,y,parameters)
print "[+]Cost:" + str(cost)
print "[+]Time:%ds"%(int(t2-t1))

plt.figure(1)
plt.subplot(211)
plt.plot(costList)
plt.subplot(212)
plt.plot(aList)
plt.show()


