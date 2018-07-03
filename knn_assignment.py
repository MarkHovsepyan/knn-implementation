import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import operator

data = pd.read_csv('gender_data.csv')

## random non-overlapping distribution (80/10/10)
np.random.seed(1998)
permute = np.random.permutation(data.index)
size = len(data)
train_end = int(0.8 * size)
test_end = int(0.1 * size) + train_end
train = data.ix[permute[:train_end]]
test = data.ix[permute[train_end:test_end]]
validate = data.ix[permute[test_end:]]

train = pd.DataFrame.as_matrix(train)
validate = pd.DataFrame.as_matrix(validate)
test = pd.DataFrame.as_matrix(test)


predictions = []  # global variable for predictions


def euclideanDist(inst1, inst2):
    distance = 0
    for x in range(1, 3):
        distance += pow((inst1[x] - inst2[x]), 2)
    return math.sqrt(distance)


def fit(trainSet, testInst, k):
    distances = []
    neighbors = []
    for x in range(len(trainSet)):
        dist = euclideanDist(testInst, trainSet[x])
        distances.append((trainSet[x], dist))
    distances.sort(key = operator.itemgetter(1))
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def predict(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][0]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key = operator.itemgetter(1), reverse = True)
    return sortedVotes[0][0]

def knn_results(k):
    for x in range(len(test)):
        for y in range(1, 3):
            train[x][y] = float(train[x][y])
            test[x][y] = float(test[x][y])
            validate[x][y] = float(validate[x][y])

        neighbors = fit(train, test[x], k)
        result = predict(neighbors)
        predictions.append(result)
        #print('> predicted = ' + repr(result) + ', actual = ' + repr(test[x][0]))


## call to print results for any k
knn_results(15) # for example: 15

## using test set
def accuracy(trainSet, testSet, k):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][0] is predict(fit(trainSet, testSet[x], k)):
            correct += 1
    return (correct / float(len(testSet)) * 100)

#print(accuracy(train, test, 15))

k_set = [2, 3, 5, 8, 10, 15, 20, 25, 30]

## using validation set
def choose_k(trainSet, validateSet):
    my_k = 0
    my_accuracy = 0
    for k in k_set:
        if my_accuracy < accuracy(trainSet, validateSet, k):
            my_accuracy = accuracy(trainSet, validateSet, k)
            my_k = k
    return my_k

#print(choose_k(train, validate))
## This method has really bad time complexity (extremely bad) :(
## Though it does work