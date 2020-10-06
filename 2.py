from math import sqrt
import numpy as np
from numpy.core.fromnumeric import sort

points = np.array([[4, 1, "o"], [3, 2, "o"], [2, 3, "o"], [1, 3, "+"], [2, 4, "+"], [1, 5, "+"]])

def main():
    #cross validate and get error for each fold
    pt_len = len(points)
    #init cv error values
    cverr1 = 0
    cverr3 = 0

    for i in range(pt_len):
        #split data set into one test and the rest training sets
        print(f"Fold {i+1}: ")
        test = np.array([points[i]])
        # print(f"test set: {test}\n")
        train = np.append(points[:i], points[i+1:], axis=0)
        # print(f"train set:\n{train}\n")

        #calculate distances between test point and training points
        distMatrix = calcDists(train, test)
        newtrain = np.concatenate([train, distMatrix], axis=1)
        #sort distances smallest to largest and get index array
        sTrain = newtrain[newtrain[:,3].argsort()]
        # print(sTrain)


        #1NN classification
        oneNNclass = classifyKnn(sTrain, 1)
        #3NN classification
        threeNNclass = classifyKnn(sTrain, 3)

        folderr1 = 0
        folderr3 = 0
        #1NN error calc
        if oneNNclass == test[0,2]:
            # print("success")
            folderr1 += 0
        else:
            # print("fail")
            folderr1 += 1
        #3NN error calc
        if threeNNclass == test[0,2]:
            # print("success")
            folderr3 = 0
        else:
            # print("fail")
            folderr3 = 1

        print(f"1NN fold {i+1} error: {folderr1}")
        print(f"3NN fold {i+1} error: {folderr3}\n")
        cverr1 += folderr1
        cverr3 += folderr3
    cverr1 /= pt_len
    cverr3 /= pt_len
    print(f"Averaged 1NN CVerror: {cverr1}")
    print(f"Averaged 3NN CVerror: {cverr3}")

def classifyKnn(_sTrain, _k):
    #takes test set and sorted training set and classifies all test data based on k-nearest-neighbors
    o = 0
    p = 0
    data = _sTrain[:_k,2:3]

    for i, train in zip(range(len(data)), data):
        if train[0] == 'o':
            o = o + 1
        elif train[0] == '+':
            p = p + 1
        else:
            print(f"training point {i} missing classification")
            prtKnn(_sTrain, len(_sTrain))

    if o > p: return 'o'
    elif o < p: return '+'
    else: return 'tie'

# def sortTrain(_trainset, _distances):
#     #sort the distances array usurped by sTrain = newtrain[newtrain[:,3].argsort()]
#     sorteddists = np.sort(_distances, axis=0, kind='mergesort')
#     sortedtrainset = []
#     for value in sorteddists:
#         #build sorted training set with closest point first, furthest point last
#         index = np.where(_distances == value)[0][0]
#         sortedtrainset.append(_trainset[index])

#     newtrain_ = np.concatenate([sortedtrainset, sorteddists.reshape(5,1)], axis=1)
#     return newtrain_

def prtKnn(_sTrainSet, _k):
    #given an array of training data and their respective distances from the test point, print the k nearest neighbors
    printSet = _sTrainSet[:_k,:]
    print(f"{_k} nearest neighbors:")
    for i in range(len(printSet)):
        #print k nearest neighbor points
        print(f"{i+1} nearest: {printSet[i,:3]}, {printSet[i,3]}")

def calcDists(_train, _test):
    #takes test set and training set and returns distance matrix where entries are train samples and column values are the distance between that training sample and a test sample
    tr_len = len(_train)
    te_len = len(_test)
    distMatrix_ = np.empty((tr_len, 0), dtype="float16")
    for i, test in zip(range(te_len), _test):

        #for each member of the test set
        dists = np.array([], dtype="float16")
        for j, train in zip(range(tr_len), _train):
            #get distances between test point and remaining points
            summation = 0
            for dimension in range(2):
                #calculate sum of the difference across all dimensions
                # print(test[dimension])
                summation = summation + (int(train[dimension]) - int(test[dimension]))**2
            #calc final distance
            dist = round(sqrt(summation), 2)
            # print(f"distance between test{i} and train{j}: {dist}")
            #append dist to array
            dists = np.append(dists, dist)
        distMatrix_ = np.concatenate([distMatrix_, dists.reshape(len(_train),1)], axis=1)

    return distMatrix_

if __name__ == "__main__":
    main()