'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
'''
import numpy as np

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

def LoadDataset():
    Train_set, Trainlable_set, Test_set, Testlable_set = [], [], [], []
    with open('horseColicTraining.txt','r') as fTrain:
        for line in fTrain:
            currline = line.strip().split('\t')
            lineArr = [float(item) for item in currline[:21]]
            Train_set.append(lineArr)
            Trainlable_set.append(float(currline[21]))
    with open('horseColicTest.txt','r') as fTest:
        for line in fTest:
            currline = line.strip().split('\t')
            lineArr = [float(item) for item in currline[:21]]
            Test_set.append(lineArr)
            Testlable_set.append(float(currline[21]))
    return Train_set, Trainlable_set, Test_set, Testlable_set

def stocGradAscent(X, Y, numIter = 150):
    m, n = X.shape
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.0001
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            a = sigmoid(sum(X[randIndex] * weights))
            error = Y[randIndex] - a
            weights = weights + alpha * error * X[randIndex]
            del(dataIndex[randIndex])
    return weights

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    trainingSet, trainingLabels, testSet, testLables = LoadDataset()
    trainWeights = stocGradAscent(np.array(trainingSet), trainingLabels, 1000)
    m, n = np.array(testSet).shape
    counterror = 0
    for i in range(m):
        if(classifyVector(testSet[i], trainWeights) != testLables[i]):
            counterror += 1
    errorRate = float(counterror / m)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))

if __name__ == '__main__':
    multiTest()