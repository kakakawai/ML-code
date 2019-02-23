import scipy.io as sio
import numpy as np


def LoadData(name):
    print "[+]Loading Data"
    data = sio.loadmat(name)
    #weights = sio.loadmat('ex4weights.mat')

    #print np.shape(data['X'])
    #print np.zeros(np.shape(weights['Theta1']))
    #print np.shape(data['y'])
    #print data['y']
    #print np.shape(weights['Theta1'])
    #print np.shape(weights['Theta2'])
    #print "[+]Loading Data ....OK"
    print data['Theta1']
    print data['Theta2']
    print np.shape(data['Theta1'])
    print np.shape(data['Theta2'])
    return data['Theta1'],data['Theta2']

LoadData("Theta-2017-08-09-19-23-17.mat")