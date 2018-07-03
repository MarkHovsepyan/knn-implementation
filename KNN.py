import csv
import math
import operator
import random


def loadDataset(data, split, trainingSet, testSet):
    with open(data) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
    for x in range(len(dataset) - 1):
        for y in range(4):
            dataset[x][y] = float(dataset[x][y])
        if random.random() < split:
            trainingSet.append(dataset[x])
        else:
            testSet.append(dataset[x])


def euclideanDist(inst1, inst2, length):
    distance = 0
    for x in range(length):
        distance += pow((inst1[x] - inst2[x]), 2)
    return math.sqrt(distance)


def getNeighbors(trainSet, testInst, k):
    distances = []
    length = len(testInst) - 1
    for x in range(len(trainSet)):
        dist = euclideanDist(testInst, trainSet[x], length)
        distances.append((trainSet[x], dist))
    distances.sort(key = operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key = operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def main():
    trainSet = []
    testSet = []
    split = 0.7
    loadDataset('iris.csv', split, trainSet, testSet)
    print('Train set: ' + repr(len(trainSet)))
    print('Test set: ' + repr(len(testSet)))

    # generate predictions
    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted = ' + repr(result) + ', actual = ' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')


# call main
main()