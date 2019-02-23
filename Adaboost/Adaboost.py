# -*- coding:utf-8 -*-
import numpy as np

def loadSimpData():
    datMat = np.array([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def simpleClassify(data,dimen,threshVal,threshIneq):
    retArray = np.ones((np.shape(data)[0],1))
    if threshIneq == 'lt':
        retArray[data[:,dimen] <= threshVal] = -1.0
    else:
        retArray[data[:,dimen] > threshVal] = -1.0
    return retArray

def findBestClassifyer(data,labels,weight):
    m,n = np.shape(data)
    numSteps = 10.0
    bestClassifyer = {}
    bestPredictResult = np.zeros((m,1))
    minError = np.inf
    for i in range(n):#ergodic features
        valMin = np.min(data[:,i])
        valMax = np.max(data[:,i])
        setpSize = float(valMax-valMin)/numSteps
        for j in range(-1,int(numSteps)+1):#从范围之外遍历阈值
            for inequal in {"lt","gt"}:#大于或小于阈值
                threshVal = (valMin + float(j)*setpSize)
                predictResult = simpleClassify(data,i,threshVal,inequal)
                errLabels = np.ones((m,1))
                errLabels[predictResult == labels] = 0
                weightedError = np.dot(weight.T,errLabels)
                #print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)

                if weightedError < minError:
                    minError = weightedError
                    bestPredictResult = np.copy(predictResult)
                    bestClassifyer["dim"] = i
                    bestClassifyer["thresh"] = threshVal
                    bestClassifyer["ineq"] = inequal
    return bestClassifyer,minError,bestPredictResult

def adaBoostTrain(data,labels,iternum):
    weakClass = []
    m = np.shape(data)[0]
    weight = np.ones((m,1))/m
    allClassEst = np.zeros((m,1))

    for iter in range(iternum):
        allErrors = np.ones((m,1))
        bestClassifyer,error,predictResult = findBestClassifyer(data,labels,weight)
        #print "[%d]Weight:%s"%(iter,str(weight.T))
        alpha = float(0.5*np.log((1.0-error)/np.max(error,1e-16)))
        bestClassifyer["alpha"] = alpha
        weakClass.append(bestClassifyer)
        #print "[%d]Predict:%s"%(iter,str(predictResult.T))

        expon = -1*alpha*labels*predictResult
        weight = weight*np.exp(expon)
        weight = weight/weight.sum()

        allClassEst += alpha*predictResult
        #print "[%d]allClassEst:%s"%(iter,str(allClassEst.T))

        allErrors[np.sign(allClassEst) == labels] = 0
        Accuary = 1- float(allErrors.sum())/m
        #print "[%d]Accuary:%s"%(iter,str(Accuary))
        if Accuary == 1.0:break
    return weakClass

def predict(data,classifiers):
    m = np.shape(data)[0]
    allPredict = np.zeros((m,1))
    for i in range(len(classifiers)):
        predict = simpleClassify(data,classifiers[i]["dim"],classifiers[i]["thresh"],classifiers[i]["ineq"])
        allPredict += (classifiers[i]["alpha"] * predict)
        print allPredict
    return np.sign(allPredict)

if __name__ == "__main__":
    weight = np.ones((5,1))/5
    data,labels = loadSimpData()
    labels = np.array(labels).reshape((5,1))
    #print np.shape(labels)
    classifiers = adaBoostTrain(data,labels,9)
    data = np.array([0,0]).reshape((1,2))
    data = np.array([[5,5],[0,0]])
    print predict(data,classifiers)

