from sklearn.tree import DecisionTreeClassifier
import csv


def compute(dataSet, dataRes, dataTest, dataTestID):
    dtc = DecisionTreeClassifier(criterion='entropy', max_depth=6)
    dtc.fit(dataSet, dataRes)
    with open('decision_tree.csv', 'w') as subFile:
        fileWriter = csv.writer(subFile, delimiter=',')
        fileWriter.writerow(['PassengerID', 'Survived'])

        for i, row in enumerate(dataTest):
            predict = dtc.predict([row])[0]
            fileWriter.writerow([dataTestID[i], predict])
    print('Tree', dtc.score(dataSet, dataRes))
