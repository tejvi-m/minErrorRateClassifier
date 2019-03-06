import pandas as pd
import numpy as np
from multivariateNormal import *

accuracies = []
accs1 = []
accs2 = []

#add column names
features = []

data1 = pd.read_csv("./Data/class1.csv")
data2 = pd.read_csv("./Data/class2.csv")

data1 = data1[features]
data2 = data2[features]

split1 = 14
split2 = 15


for i in range(100):
    data1 = data1.sample(frac=1).reset_index(drop=True)
    train1 = data1[0:split1]
    data2 = data2.sample(frac=1).reset_index(drop=True)
    train2 = data2[0:split2]

    test1 = data1[features][split1:]
    test2 = data2[features][split2:]

    right1 = 0
    for row in test1.values:
        prob1, prob2 = getProbabilities(row, train1, train2)
        if(action(prob1, prob2) == 1):
            right1 += 1

    right2 = 0
    for row in test2.values:
        prob1, prob2 = getProbabilities(row, train1, train2)

        if(action(prob1, prob2) == 0):
            right2 += 1

    accuracy = (right1 + right2) / (len(test1.iloc[0].values) + len(test2.iloc[0].values))
    acc1 = right1/len(test1.iloc[0].values)
    acc2 = right2/len(test2.iloc[0].values)
    print("acc: ", accuracy * 100, "\tacc1: ", acc1 * 100,
        "\tacc2: ", acc2 * 100)
    accuracies.append(accuracy)
    accs1.append(acc1)
    accs2.append(acc2)


print("mean_acc: ", np.mean(accuracies) * 100)
print("min_acc: ", min(accuracies) * 100, "\tmax_acc: ", max(accuracies) * 100)
print("class wise accuracies: ")
print("mean_acc1: ", np.mean(accs1) * 100)
print("min_acc1: ", min(accs1) * 100, "\tmax_acc1: ", max(accs1) * 100)
print("mean_acc2: ", np.mean(accs2) * 100)
print("min_acc2: ", min(accs2) * 100, "\tmax_acc2: ", max(accs2) * 100)
