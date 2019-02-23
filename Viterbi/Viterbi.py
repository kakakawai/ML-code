# -*- coding:utf-8 -*-
import numpy as np
"""
==========Inputs============
Observation space O(1,N)
State space S(1,K)

Observation sequence Y(1,T)

Transition matrix A(K,K)
Emission matrix B(K,N)
Initial Probabilities Pi(1,K)
"""
"""
==========Outputs===========
The most likely hidden state sequence X(1,T)
"""

"""
===========Temp=============
T1(K,T)：保存序列位置为t时，状态为s的最优路径的概率：
T2(K,T)：保存序列位置为t时，状态为s的最优路径的上一个状态
"""

def Viterbi(O,S,Y,A,B,Pi):
    K = len(S)
    T = len(Y)
    T1 = np.zeros((K,T))
    T2 = np.zeros((K,T))
    Z = np.zeros((1,T))
    X = np.zeros((1,T))

    """forward first init"""
    T1[:,0] = (Pi*B[:,Y[0]]).reshape(K)                #初始状态概率矩阵和观察概率矩阵相乘，作为第t=0列的值，表示第1个位置上，x取不同状态state时的到第一个观察值Y[0]的概率
    T2[:,0] = None                                     #初始状态的上一个状态为None

    """forward"""
    for t in range(1,T):                              #从t=1开始遍历观察序列
        for state in range(K):                        #对序列每一个位置t的隐变量的状态进行估计
            temp = T1[:,t-1]*A[:,state]*B[state,Y[t]]  #计算从上一个序列位置t-1到当前序列位置t且隐变量x取值为state时的概率
            T1[state, t] = np.max(temp)                #取概率最大值作为从上一个序列位置t-1到当前序列位置t的最优路径
            T2[state, t] = np.argmax(temp)            #记录最优路径的是由上一个位置的那个state计算得到，便于backward时能够查找

    """get the most likely end state"""
    Z[0,-1] = np.max(T1[:, -1])                       #取最后一个位置t=T-1的概率最大值，则该节点对应的路径为最可能的隐变量序列
    X[0, -1] = np.argmax(T1[:, -1])                   #同样，记录上一个序列位置的state值

    """backward"""
    for t in reversed(range(1,T)):
        X[0, t - 1] = T2[X[0, t], t]
        #Z[0, t - 1] = T1[X[0, t - 1], t - 1]

    return X



if __name__ == '__main__':
    #'''
    obs = ('Red', 'White')
    states = ('A', 'B', 'C')
    start_p = np.array([0.2, 0.4, 0.4], dtype="float64")
    trans_p = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    emit_p = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
    Y = np.array([0, 1, 0])
    '''
    obs = ('normal', 'cold', 'dizzy')
    states = ('Healthy', 'Fever')
    start_p = np.array([0.6, 0.4], dtype="float64")
    trans_p = np.array([[0.7, 0.3], [0.4, 0.6]])
    emit_p = np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])
    Y = np.array([0, 1, 2])
    '''

    result = Viterbi(obs,states,Y,trans_p,emit_p,start_p)
    print result

