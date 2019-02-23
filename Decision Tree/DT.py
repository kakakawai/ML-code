# -*- coding:utf-8 -*-
import math
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

def getEntropy(data):
    numEntries = len(data)
    labelCounts = {}
    entropy = 0.0

    for item in data:
        currentLabel = item[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] +=1

    for key in labelCounts:
        temp = float(labelCounts[key]) / numEntries
        entropy -= temp * math.log(temp,2)

    return entropy

def chooseBestFeature(dataSet):
    numFeature = len(dataSet[0]) -1
    baseEntropy = getEntropy(dataSet)
    bestGain = 0.0
    bestFeature = -1
    for i in range(numFeature):
        newEntropy = 0.0
        featList = [item[i] for item in dataSet]
        uniqueVals = set(featList)
        for value in uniqueVals:
            subDataSet = spliteDataSet(dataSet,i,value)
            temp = float(len(subDataSet))/len(dataSet)
            newEntropy += temp * getEntropy(subDataSet)
        Gain = baseEntropy - newEntropy
        if Gain > bestGain:
            bestGain = Gain
            bestFeature = i
    return bestFeature

def spliteDataSet(data,index,value):
    subDataSet = []
    tempSet = []
    for item in data:
        if item[index] == value:
            tempSet = item[:index]
            tempSet.extend(item[index+1:])
            subDataSet.append(tempSet)

    return subDataSet

def majoritycnt(classList):
    classCount={}
    for vote in classList:
        if classCount[vote] not in classList:
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


def creatTree(data,labels):
    classList = [example[-1] for example in data]
    if classList.count(classList[0]) == len(data):
        return classList[0]
    if len(data[0]) == 1:
        return majoritycnt(classList)
    bestFeature = chooseBestFeature(data)
    bestFeatureLabel = labels[bestFeature]
    Tree = {bestFeatureLabel:{}}
    del(labels[bestFeature])
    featureValues = [item[bestFeature] for item in data]
    uniqueVals = set(featureValues)
    for value in uniqueVals:
        subLabels = labels[:]
        Tree[bestFeatureLabel][value] = creatTree(spliteDataSet(data,bestFeature,value),subLabels)

    return Tree

def classify(tree,labels,inputVec):
    nowDict = tree.keys()[0]
    nowDictIndex = labels.index(nowDict)
    nextDict = tree[nowDict]
    for value in nextDict.keys():
        if inputVec[nowDictIndex] == value:
            if type(nextDict[value]).__name__=='dict':
                classLabel = classify(nextDict[value],labels,inputVec)
            else:
                classLabel = nextDict[value]
    return classLabel

def getTreeDepth(Tree):
    Depth = 0
    nowDict = Tree.keys()[0]
    secondDict = Tree[nowDict]
    for value in secondDict.keys():
        thisDepth = 0
        if type(secondDict[value]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[value])
        else:
            thisDepth +=1
        if thisDepth > Depth:
            Depth = thisDepth
    return Depth

def getNumLeafs(Tree):
    numLeafs = 0
    nowDict = Tree.keys()[0]
    secondDict = Tree[nowDict]
    for value in secondDict.keys():
        if type(secondDict[value]).__name__ =='dict':
            numLeafs += getNumLeafs(secondDict[value])
        else:
            numLeafs += 1
    return numLeafs

if __name__ == "__main__":
    data,labels = createDataSet()
    label = labels[:]
    #print getEntropy(data)
    #print spliteDataSet(data,0,1)
    #print spliteDataSet(data,0,0)
    #print chooseBestFeature(data)
    myTree = creatTree(data,labels)
    print getTreeDepth(myTree)
    print getNumLeafs(myTree)
    print classify(myTree,label,[1,0])
    print classify(myTree, label, [1, 1])