import csv
import random
import math
import operator
from sklearn.cross_validation import KFold
from sklearn import cross_validation

def main():
    trainSet=[]
    testSet=[]
    avgaccuracy=0
    n_folds= int(raw_input('enter the n_folds value:'))
    k =  int(raw_input('enter the k value:'))
    fina=raw_input('enter the filename :')        
    loadData(fina,n_folds,trainSet, testSet)
    answers=[]
    for i in range(n_folds):
        for x in range(len(testSet[i])):
            Kneighbors = getValueK(trainSet[i], testSet[i][x], k)
            result = Response(Kneighbors)
            answers.append(result)
            test=testSet[i]
        accuracy = Accuracy(testSet[i], answers)
        avgaccuracy+=accuracy/n_folds
        del answers[:]
    print 'Avg Accuracy: ' + str(avgaccuracy)

 
def loadData(filename,n_folds,trainingSet=[] , testSet=[]):
    l=0
    train=[]
    test=[]
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(len(dataset[0])-1):
                dataset[x][y] = float(dataset[x][y])
        kf = KFold(len(dataset), n_folds, shuffle =True)
        for train_index, test_index in kf:
            for i in train_index:
                train.append(dataset[i][:])
            for i in test_index:
                test.append(dataset[i][:])
            trainingSet.append(list(train))
            testSet.append(list(test))
            del train[:]
            del test[:]
            l+=1

 
 
def euclidean(occurrence1, occurrence2, length):
    dist = 0
    for x in range(length):
        dist += pow((occurrence1[x] - occurrence2[x]), 2)
    return math.sqrt(dist)
 
def getValueK(trainSet, testOccurrence, k):
    distances = []
    length = len(testOccurrence)-1
    for x in range(len(trainSet)):
        dist = euclidean(testOccurrence, trainSet[x], length)
        distances.append((trainSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    Kneighbors = []
    for x in range(k):
        Kneighbors.append(distances[x][0])
    return Kneighbors
 
def Response(Kneighbors):
    classpredicsts = {}
    for x in range(len(Kneighbors)):
        response = Kneighbors[x][-1]
        if response in classpredicsts:
            classpredicsts[response] += 1
        else:
            classpredicsts[response] = 1
    finalPredicts = sorted(classpredicsts.iteritems(), key=operator.itemgetter(1), reverse=True)
    return finalPredicts[0][0]
 
def Accuracy(testSet, answers):
    counts = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == answers[x]:
            counts += 1
    return (counts/float(len(testSet))) * 100.0

main()

#REFERENCE
#http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
