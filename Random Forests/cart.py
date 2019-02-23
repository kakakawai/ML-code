# -*- coding:utf-8 -*-
import numpy as np
import operator
import copy

def getData():
    data = np.array([
        [0,0,0,0,0],
        [0,0,0,1,0],
        [0,1,0,1,1],
        [0,1,1,0,1],
        [0,0,0,0,0],
        [1,0,0,0,0],
        [1,0,0,1,0],
        [1,1,1,1,1],
        [1,0,1,2,1],
        [1,0,1,2,1],
        [2,0,1,2,1],
        [2,0,1,1,1],
        [2,1,0,1,1],
        [2,1,0,2,1],
        [2,0,0,0,0]
    ])
    labels = np.array(["Year","Work","House","Credit","label"])
    return data,labels

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

def Gini(D):
    labels = [item[-1] for item in D]#D[:,-1]
    labelSet = set(labels)
    GiniRes = 0.0

    for label in labelSet:
        count = np.count_nonzero(labels == label)
        p = float(count)/len(D)
        GiniRes += p**2

    return 1-GiniRes

def FeatureGini(D,leftTree,rightTree):
    Pleft = float(len(leftTree))/len(D)
    Pright = float(len(rightTree))/len(D)
    FGini = Pleft * Gini(leftTree) + Pright * Gini(rightTree)

    return FGini

def choose(D):
    numFeature = len(D[0])-1
    bestGini = 128
    bestFeature = -1
    bestVal = None
    for i in range(numFeature):
        featureValues = [item[i] for item in D]#D[:,i]
        valSet = set(featureValues)
        for value in valSet:
            testD = D[:]
            leftTree,rightTree = spliteTree(testD,i,value)
            nowGini = FeatureGini(testD,leftTree,rightTree)
            if nowGini < bestGini:
                bestGini = nowGini
                bestFeature = i
                bestVal = value

    return bestFeature,bestVal


def spliteTree(D,feature,value):
    leftTree = []
    rightTree = []
    for item in D:
        if item[feature] == value:
            leftTree.append(np.append(item[:feature],item[feature+1:]))
        else:
            rightTree.append(np.append(item[:feature], item[feature + 1:]))
    return np.array(leftTree),np.array(rightTree)

def majoritycnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount:
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def isTree(obj):
    return (type(obj).__name__=="dict")

def prune(tree,data,testdata):
    treeLeft = tree["left"]
    treeRight = tree["right"]


    if testdata.shape[0]==0:return tree
    if isTree(treeRight) or isTree(treeLeft):
        subLeftTest,subRightTest = spliteTree(testdata,tree['Feature'],tree["value"])
        subLeft, subRight = spliteTree(data, tree['Feature'], tree["value"])
    if isTree(treeRight): treeRight = prune(treeRight,subRight,subRightTest)
    if isTree(treeLeft): treeLeft = prune(treeLeft,subLeft,subLeftTest)
    if not isTree(treeRight) and not isTree(treeLeft):
        subLeftTest,subRightTest = spliteTree(testdata,tree['Feature'],tree["value"])
        classList = [item[-1] for item in data]
        print classList
        nodeClass = majoritycnt(classList)


        errorNCount = 0
        errorCount = 0
        for item in testdata:
            errorNCount += 1 if item[-1] != nodeClass else 0
            errorCount += 1 if item[-1] != classify(tree, item) else 0

        if errorCount > errorNCount:
            return nodeClass
    return tree


def creat(D,minSampleNum,minGini):
    Tree = {}
    #if not len(D):return Tree
    if not len(D): return None
    classList = [item[-1] for item in D]#list(D[:,-1])
    if len(classList) and classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(D[0]) == 1 or len(D) < minSampleNum+1 :# or Gini(D) < minGini :
        return majoritycnt(classList)
    bestFeature,bestValue = choose(D)
    leftTree,rightTree = spliteTree(D,bestFeature,bestValue)
    leftTree = creat(leftTree,minSampleNum,minGini)
    rightTree = creat(rightTree,minSampleNum,minGini)
    Tree["value"] = bestValue
    Tree["Feature"] = bestFeature
    Tree["left"] = leftTree
    Tree["right"] = rightTree
    return Tree

def classify(tree,data):
    if data[tree["Feature"]] == tree["value"]:
        if type(tree["left"]).__name__=="dict":
            classLabel = classify(tree["left"],data)
        else:
            classLabel = tree["left"]
    if data[tree["Feature"]] != tree["value"]:
        if type(tree["right"]).__name__=="dict":
            classLabel = classify(tree["right"],data)
        else:
            classLabel = tree["right"]

    return classLabel

if __name__ =="__main__":
    #data= loadDataSet("ex0.txt")
    #data = np.array(data)
    data, _ = getData()
    print data
    tree = creat(data,2,1000)
    print tree
    label = classify(tree,[0,0,0,0])
    print label

