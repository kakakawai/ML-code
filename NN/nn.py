#-*- coding:utf-8 -*-
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.optimize as opt
from math import exp
import random
import time

input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;          # 10 labels, from 1 to 10



def NNCostFunc(X,y,Theta1,Theta2,Lambda,num_labels):
    m = X.shape[0]
    y_v = np.zeros((m,num_labels))
    #print np.shape(y_v)
    J = 0.0
    Theta1_grad = np.zeros(np.shape(Theta1))
    Theta2_grad = np.zeros(np.shape(Theta2))
    '''(A .* B).T == B.T .* A.T'''
    '''
    A1 = A.T
    W1 = W.T
    A1 .* W1.T == A.T .* W == (W.T .* A).T
    '''
    a1 = np.column_stack((np.ones((m,1)),X))
    #print np.shape(a1)
    z2 = a1.dot(Theta1.T)
    #print np.shape(z2)
    a2 = sigmoid(z2)
    #print np.shape(a2)
    a2 = np.column_stack((np.ones((m,1)),a2))
    #print np.shape(a2)
    hx = sigmoid(a2.dot(Theta2.T))
    for i in range(0,m):
        y_v[i,int(y[i]-1)] = 1#这边-1后所有label向前移一位，所以计算结果是要记得label+1
    #print y_v
    for i in range(0,m):
        J += ((-y_v[i].dot(np.log(hx[i].T))) - (1-y_v[i]).dot(np.log(1-hx[i].T)))
    J = J/m
    J += (Lambda/(2.0*m))*(np.sum(Theta1[:,1:]**2)+np.sum(Theta2[:,1:]**2))
    #print J

    Delta1 = np.zeros(np.shape(Theta1))
    Delta2 = np.zeros(np.shape(Theta2))

    delta3 = hx - y_v
    delta2 = (delta3.dot(Theta2[:,1:])*sigmoidGradient(z2))

    Delta1 = delta2.T.dot(a1)
    Delta2 = delta3.T.dot(a2)

    Theta1_grad = Delta1/m
    Theta2_grad = Delta2/m

    Theta1_grad[:,1:] = Theta1_grad[:,1:] + (Lambda/m)*Theta1[:,1:]
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (Lambda / m) * Theta2[:, 1:]

    #print np.shape(Theta1_grad)
    #print Theta1_grad
    #print np.shape(Theta2_grad)
    #print Theta2_grad

    return J,Theta1_grad,Theta2_grad


def LoadData():
    print "[+]Loading Data"
    data = sio.loadmat("ex4data1.mat")
    weights = sio.loadmat('ex4weights.mat')
    print np.shape(data['X'])
    #print np.zeros(np.shape(weights['Theta1']))
    #print np.shape(data['y'])
    #print data['y']
    print np.shape(weights['Theta1'])
    #print np.shape(weights['Theta2'])
    #print "[+]Loading Data ....OK"

    return data['X'],data['y'],weights['Theta1'],weights['Theta2']

def sigmoid(z):
    g = 1.0 / (1.0 + np.exp(-z))
    return g

def sigmoidGradient(z):
    g = sigmoid(z)
    gD = g*(1-g)
    return gD

def RandomInitWeights(LayIn,LayOut):
    Epsilon = 0.12
    W = np.random.random((1+LayIn, LayOut)) * 2 * Epsilon - Epsilon
    return W

def CheckNNGrad(Lambda):
    inputLaySize = 3
    hiddenLaySize = 5
    numLabel = 3
    m = 5
    testTheta1 = RandomInitWeights(inputLaySize+1,hiddenLaySize).T
    testTheta2 = RandomInitWeights(hiddenLaySize+1,numLabel).T
    #print np.shape(testTheta1)
    #print np.shape(testTheta2)
    X = RandomInitWeights(m,inputLaySize)
    y = np.zeros([m,1])
    #print np.shape(X)
    #print np.shape(y)
    for i in range(m):
        y[i,0] = random.randint(1,numLabel)
    #print y

    Cost,Theta1Grad,Theta2Grad = NNCostFunc(X,y,testTheta1,testTheta2,Lambda,numLabel)
    num1Grad,num2Grad = ComputeNumericalGradient(NNCostFunc,X,y,testTheta1,testTheta2,Lambda,numLabel)
    print Theta1Grad - num1Grad
    #print num1Grad
    print Theta2Grad - num2Grad
    #print num2Grad

def ComputeNumericalGradient(costfunc,X,y,theta1,theta2,Lambda,numlabels):#
    num1Grad = np.zeros(np.shape(theta1))
    num2Grad = np.zeros(np.shape(theta2))
    perturb1 = np.zeros(np.shape(theta1))
    perturb2 = np.zeros(np.shape(theta2))
    e = 0.0001

    for i in range(theta1.shape[0]):
        for j in range(theta1.shape[1]):
            perturb1[i,j] = e
            cost1,loss1,loss = costfunc(X,y,theta1-perturb1,theta2,Lambda,numlabels)
            cost2,loss2,loss = costfunc(X,y,theta1+perturb1,theta2,Lambda,numlabels)
            num1Grad[i,j] = (loss2[i,j]-loss1[i,j])/(2.0*e)
            perturb1[i, j] = 0
    #print num1Grad

    for i in range(theta2.shape[0]):
        for j in range(theta2.shape[1]):
            perturb2[i, j] = e
            cost1,loss,loss1 = costfunc(X, y, theta1, theta2 - perturb2, Lambda, numlabels)
            cost2,loss,loss2 = costfunc(X, y, theta1, theta2 + perturb2, Lambda, numlabels)
            num2Grad[i, j] = (loss2[i,j] - loss1[i,j]) / (2.0 * e)
            perturb2[i, j] = 0
    #print num2Grad

    return num1Grad,num2Grad


def costFunc(Theta):
    Theta1 = np.reshape(Theta[0:hidden_layer_size * (1 + input_layer_size)],(1 + input_layer_size, hidden_layer_size)).T
    Theta2 = np.reshape(Theta[hidden_layer_size * (1 + input_layer_size):], (hidden_layer_size + 1, num_labels)).T
    return NNCostFunc(X,y,Theta1,Theta2,Lambda,num_labels)

'''合并Theta列向量'''
def getTheta(Theta1,Theta2):
    Theta11 = Theta1.T.ravel()
    Theta22 = Theta2.T.ravel()
    Theta = np.concatenate((Theta11, Theta22), axis=0)
    Theta.shape = (-1, 1)
    return Theta

def predict(Theta1,Theta2,X,y):
    m = X.shape[0]
    p = np.zeros((m,1))

    a1 = np.column_stack((np.ones((m, 1)), X))
    z2 = a1.dot(Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.column_stack((np.ones((m, 1)), a2))
    hx = sigmoid(a2.dot(Theta2.T))
    p = hx.argmax(axis=1)
    p = p+1
    p.shape = (-1,1)
    Accuracy = 1 - (np.count_nonzero(y - p) / (1.0 * m))
    print '[+]Accuracy:%f'%(Accuracy)
    return Accuracy

J_list = []
a_list = []
X,y,Theta1,Theta2 = LoadData()
predict(Theta1,Theta2,X,y)

Lambda = 1
InitTheta1 = RandomInitWeights(input_layer_size,hidden_layer_size).T
InitTheta2 = RandomInitWeights(hidden_layer_size,num_labels).T

alpha = 0.5
decay_rate = 0.99
t1 = time.time()
for i in range(1000):
    J,Theta1grad,Theta2grad = NNCostFunc(X, y, InitTheta1, InitTheta2, Lambda, num_labels)
    InitTheta1 += -alpha*Theta1grad
    InitTheta2 += -alpha*Theta2grad
    J_list.append(J)
    a_list.append(alpha)
    if not i%10:
        alpha = alpha * 0.999
    print "\r[+]%d/1000\r"%(i)
t2 = time.time()

predict(InitTheta1,InitTheta2,X,y)
print '[+]J=%f'%(J)
print "[+]Time:%ds"%(int(t2-t1))

#plot
plt.figure(1)
plt.subplot(211)
plt.plot(J_list)
plt.subplot(212)
plt.plot(a_list)
plt.show()

'''
nowTime = str(time.strftime("%Y-%m-%d-%H-%M-%S"))
fileName = 'Theta-'+nowTime+'.mat'
print '[+]Saving Theta'
sio.savemat(fileName, {'Theta1': InitTheta1,'Theta2': InitTheta2})
print '[+]Saved!'
'''



