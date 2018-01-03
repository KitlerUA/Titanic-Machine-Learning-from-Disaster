from sklearn.neighbors import KNeighborsClassifier
import csv
import numpy


class Weight:
    def __call__(self, listOfWeights):
        return [-1.9668731298602922, 8.5148067758539732, -0.76627273720104205, -5.0393348072166884, 1.8720513966479659]


def weightFunc(list):
    return [-1.9668731298602922, 8.5148067758539732, -0.76627273720104205, -5.0393348072166884, 1.8720513966479659]


def compute(dataSet, dataRes, dataTest, dataTestID):
    instanceOfWeight = Weight()
    klaster = KNeighborsClassifier(7, algorithm='ball_tree', leaf_size=10)
    klaster.fit(dataSet, dataRes)

    with open('submission_k_nearest.csv', 'w') as subFile:
        fileWriter = csv.writer(subFile, delimiter=',')
        fileWriter.writerow(['PassengerId', 'Survived'])
        for i, row in enumerate(dataTest):
            predict = klaster.predict([row])[0]
            fileWriter.writerow([dataTestID[i], predict])
    print('K-mean', klaster.score(dataSet, dataRes))
