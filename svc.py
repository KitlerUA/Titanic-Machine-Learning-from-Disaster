from sklearn.svm import SVC
import csv


def compute(dataSet, dataRes, dataTest, dataTestID):
    svc = SVC(gamma=2, C=1)
    svc.fit(dataSet, dataRes)
    with open('svc.csv', 'w') as subFile:
        fileWriter = csv.writer(subFile, delimiter=',')
        fileWriter.writerow(['PassengerID', 'Survived'])

        for i, row in enumerate(dataTest):
            predict = svc.predict([row])[0]
            fileWriter.writerow([dataTestID[i], predict])
    print('SVC', svc.score(dataSet, dataRes))
