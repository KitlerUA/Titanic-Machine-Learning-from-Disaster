import csv
import k_nearest
import neural_network
import decision_tree
import svc
import gaussian
import random_forest

dataSet = []
dataRes = []


def sex(x):
    return {
        'male': 1,
        'female': 2,
    }[x]


def location(x):
    return {
        'C': 1,
        'Q': 2,
        'S': 3,
        '' : 2,
    }[x]


with open('train.csv', newline='\n') as trainFile:
    fileReader = csv.reader(trainFile, delimiter=',')
    next(fileReader)
    for row in fileReader:
        # class
        temp = [int(row[2])]
        # sex
        temp.append(sex(row[4]))
        # age
        if row[5] != '':
            temp.append(int(float(row[5])))
        else:
            temp.append(0)
        # brothers/sisters
        temp.append(int(row[6]))
        # children
        temp.append(int(row[7]))
        # fare
        temp.append(float(row[9]))
        # port
        temp.append(location(row[11]))
        dataSet.append(temp)

        dataRes.append(int(row[1]))


# compute average age
avg = 0
total = 0
number = 0
for row in dataSet:
    if row[2] != 0:
        total += row[2]
        number += 1
avg = int(total/number)

print(avg)

# replace zeroes with average
for row in dataSet:
    if row[2] == 0:
        if row[1] == 1:
            row[2] = avg+1
        else:
            row[2] = avg-2

dataTest = []
dataTestID = []

with open('test.csv', newline='\n') as testFile:
    fileReader = csv.reader(testFile, delimiter=',')
    next(fileReader)
    for row in fileReader:
        # class
        temp = [int(row[1])]
        # sex
        temp.append(sex(row[3]))
        # age
        if row[4] != '':
            temp.append(float(row[4]))
        else:
            temp.append(avg)
        # brothers/sisters
        temp.append(int(row[5]))
        # children
        temp.append(int(row[6]))
        # fare
        if row[8] == '':
            temp.append(40.0)
        else:
            temp.append(float(row[8]))
        # port
        temp.append(location(row[10]))
        dataTest.append(temp)
        dataTestID.append(int(row[0]))

# k_nearest.compute(dataSet, dataRes, dataTest, dataTestID)
neural_network.compute(dataSet, dataRes, dataTest, dataTestID)
# decision_tree.compute(dataSet, dataRes, dataTest, dataTestID)
# svc.compute(dataSet, dataRes, dataTest, dataTestID)
# gaussian.compute(dataSet, dataRes, dataTest, dataTestID)
# random_forest.compute(dataSet, dataRes, dataTest, dataTestID)