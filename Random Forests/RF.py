# -*- coding:utf-8 -*-
import numpy as np
import cart

def randomData(data,numData):#bagging
    #np.randomSample()
    index = np.random.choice(len(data),size=numData,replace=True)
    return data[index]

def randomFeature(data):#bootstrp
    #numpy.random.choice
    featureLen = len(data[0])-1
    k = int(np.log2(featureLen))+1
    features = list(np.random.choice(featureLen,size=k,replace=False))
    features.append(-1)
    #print features
    return data[:,features]

def randomForest(data,numTree):
    Forests = []
    for i in range(numTree):

        trainData = randomFeature(data)
        trainData = randomData(trainData,numTree)
        tree = cart.creat(trainData,2,1000)
        if tree:
            Forests.append(tree)
        #print tree
    return Forests

def predict(forests,data):
    resultList = []
    for i in range(len(forests)):
        if forests[i]:
            #print "[+]"
            #print forests[i]
            result = cart.classify(forests[i],data)
            resultList.append(result)
    #print resultList
    labelSet = set(resultList)
    labelCount = dict([(resultList.count(i),i) for i in labelSet])
    label = labelCount[max(labelCount.keys())]
    return label

def outOfBags():
    pass

def getAccuary(label,result):
    count = 0
    for i in range(len(label)):
        if label[i] == result[i]:
            count += 1
    accuary = float(count)/len(label)
    return accuary

if __name__ =="__main__":
    result = []
    data,_ = cart.getData()
    forests = randomForest(data,100)

    trainData = data[:,0:-1]
    for i in range(len(trainData)):
        label = predict(forests,trainData[i])
        result.append(label)

    acc = getAccuary(data[:,-1],result)
    print "[+]Accuary:%f" % acc