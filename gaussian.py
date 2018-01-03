from sklearn.gaussian_process import GaussianProcessClassifier
import csv


def compute(dataSet, dataRes, dataTest, dataTestID):
    gauss = GaussianProcessClassifier()
    gauss.fit(dataSet, dataRes)
    with open('gaussian.csv', 'w') as subFile:
        fileWriter = csv.writer(subFile, delimiter=',')
        fileWriter.writerow(['PassengerId', 'Survived'])

        for i, row in enumerate(dataTest):
            predict = gauss.predict([row])[0]
            fileWriter.writerow([dataTestID[i], predict])
    print('Gaussian', gauss.score(dataSet, dataRes))
