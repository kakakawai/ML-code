import numpy as np
import operator
from os import listdir

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    Distances = sqDistances ** 0.5
    sortedDistIndeicies = Distances.argsort()#Index of the array from small to large
    #sortedDistIndeicies = ((((np.tile(inX,(dataSetSize,1)) - dataSet) ** 2).sum(axis = 1)) ** 0.5).argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndeicies[i]]
        classCount[voteIlabel] =classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    f = open(filename)
    arrayOLines = f.readlines()
    LinesNum = len(arrayOLines)
    dataSet = np.zeros((LinesNum,3))
    LabelVector = []
    index = 0
    for line in arrayOLines:
        ListOfLine = line.strip().split('\t')
        dataSet[index,:] = ListOfLine[0:3]
        LabelVector.append(int(ListOfLine[-1]))
        index +=1
    return dataSet,LabelVector

def autoNorm(dataSet):
    min = dataSet.min(0)
    max = dataSet.max(0)
    ranges = max - min
    m = dataSet.shape[0]
    dataSet = dataSet - np.tile(min,(m,1))
    dataSet = dataSet / np.tile(ranges,(m,1))
    return dataSet,ranges,min

def datingClassTest(hoRatio):
    h = hoRatio
    datingDataSet, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataSet)
    m = normMat.shape[0]
    numTestVecs = int(m * h)
    errorCount = 0
    #print 'Result  Real'
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        #print '%d  %d'%(classifierResult,datingLabels[i])
        if classifierResult != datingLabels[i]:
            errorCount += 1
    errorRate = errorCount/float(numTestVecs)
    print 'Error rate:%f' % (errorRate)
    return errorRate

def img2vector(filename):
    vect = np.zeros((1,1024))
    f = open(filename)
    for i in range(32):
        lineStr = f.readline()
        for j in range(32):
            vect[0,32*i+j] = int(lineStr[j])
    return vect

def getTextLabel(filename):
    fileStr = filename.split('.')[0]
    Label = int(fileStr.split('_')[0])
    return Label

def getTrainingDataSet(filename):
    Labels = []
    fileDir = listdir(filename)
    m = len(fileDir)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        filename = fileDir[i]
        Labels.append(getTextLabel(filename))
        trainingMat[i, :] = img2vector('trainingDigits/%s' % filename)
    return trainingMat,Labels

def numberClassTest():
    TrainDataSet,TrainLabels = getTrainingDataSet('trainingDigits')
    testFileDir = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileDir)
    print 'Result  Real'
    for i in range(mTest):
        filename = testFileDir[i]
        classNumStr = getTextLabel(filename)
        testMat = img2vector('testDigits/%s' % filename)
        result = classify0(testMat,TrainDataSet,TrainLabels,3)
        print '%d  %d' %(result,classNumStr)
        if result != classNumStr:
            errorCount += 1
    errorRate = errorCount / float(mTest)
    print 'Error Count: %d'%(errorCount)
    print 'Total Count: %d'%(mTest)
    print 'Eorror Rate: %f' %(errorRate)
    return errorRate
