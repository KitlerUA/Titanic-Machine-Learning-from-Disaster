from sklearn.ensemble import RandomForestClassifier
import csv


def compute(dataSet, dataRes, dataTest, dataTestID):
    rfc = RandomForestClassifier(n_estimators=5)
    rfc.fit(dataSet, dataRes)

    with open('random_forest.csv', 'w') as subFile:
        fileWriter = csv.writer(subFile, delimiter=',')
        fileWriter.writerow(['PassengerID', 'Survived'])

        for i, row in enumerate(dataTest):
            predict = rfc.predict([row])[0]
            fileWriter.writerow([dataTestID[i], predict])
    print('Random forest', rfc.score(dataSet, dataRes))
