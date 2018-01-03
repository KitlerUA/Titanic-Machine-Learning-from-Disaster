from sklearn.neural_network import MLPClassifier
import csv
import time


currentUnixTime = lambda: int(round(time.time()))


def compute(dataSet, dataRes, dataTest, dataTestID):
    mlp = MLPClassifier(hidden_layer_sizes=(21, ), max_iter=1000, alpha=1e-5,
                        solver='sgd', verbose=0, tol=1e-5, random_state=1,
                        learning_rate='constant', learning_rate_init=.1,
                        activation='tanh')

    mlp.fit(dataSet, dataRes)
    with open('submission_nn.csv', 'w') as subFile:
        fileWriter = csv.writer(subFile, delimiter=',')
        fileWriter.writerow(['PassengerId', 'Survived'])

        for i, row in enumerate(dataTest):
            predict = mlp.predict([row])[0]
            fileWriter.writerow([dataTestID[i], predict])
    weights = [0, 0, 0, 0, 0, 0, 0]
    for i, w in enumerate(mlp.coefs_[0]):
        for j, ww in enumerate(w):
            weights[i] += ww*mlp.coefs_[1][j][0]

    print(weights)
    print('NN', mlp.score(dataSet, dataRes))
