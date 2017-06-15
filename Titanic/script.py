#import os
#import tensorflow as tf
import re
import numpy as np

# Read the data into a list
with open("train.csv", encoding="ascii") as file:
    list = file.readlines()

# First element of the list contains the feature names
features = list[0].strip().split(",")

# features that should be treated as integers
intFeatures = ["PassengerId", "Survived", "Pclass", "SibSp", "Parch"]
# features that should be treated as floats
floatFeatures = ["Age", "Fare"]

feature_values = [[] for i in range(0, len(features)) ]

# iterate over the data set in order to collect all possible feature values
for i in range(1, len(list)):
    line = re.split(",(?!\s)", list[i].strip())
    for k in range(0, len(line)):
        if features[k] in intFeatures:
            value = int(line[k])
        elif features[k] in floatFeatures:
            if line[k] != '':
                value = float(line[k])
            else:
                value = -1
        else:
            value = line[k]
        if not(value in feature_values[k]):
            feature_values[k].append(value)
            feature_values[k].sort()


# indexed version of the data set (in order to avoid strings) 
digitalized = []

for i in range(1, len(list)):
    tmpLine = []
    line = re.split(",(?!\s)", list[i].strip())
    for k in range(0, len(line)):
        if features[k] in intFeatures:
            value = int(line[k])
        elif features[k] in floatFeatures:
            if line[k] != '':
                value = float(line[k])
            else:
                value = -1
        else:
            value = line[k]
        tmpLine.append(feature_values[k].index(value))
    digitalized.append(tmpLine)


pos = [features.index("Pclass"),features.index("Sex"),features.index("Age"),features.index("SibSp"),features.index("Parch"),features.index("Ticket"),features.index("Fare"),features.index("Cabin"),features.index("Embarked")]

# input data
X = np.array(digitalized)[:, pos]
# labels
Y = np.array(digitalized)[:, features.index("Survived")]

for (x, y) in zip(X, Y):
    print(x[2], y)
