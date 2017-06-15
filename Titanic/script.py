#import os
#import tensorflow as tf
import re
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD


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


pos = [features.index("Pclass"),features.index("Sex"),features.index("Age"),features.index("SibSp"),features.index("Parch"), features.index("Fare"), features.index("Embarked")]

# input data
dataX = np.array(digitalized)[:, pos]
# labels
dataY = np.array(digitalized)[:, features.index("Survived")]

L = len(pos)
N = int(0.8 * len(dataY))
print(N)

model = Sequential()
model.add(Dense(L, input_dim=L, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['fbeta_score'])

history = model.fit(dataX, dataY, nb_epoch=10, batch_size=32, verbose=0)

for layer in model.layers:
    print(layer.get_weights())

predictions = list(map(lambda x: 1 if (x > 0.5) else 0, model.predict(dataX[N:])))
actual = dataY[N:]

# Calculate F score on the cross-validation data
tp = 0
tn = 0
fp = 0
fn = 0
for (pred, act) in zip(predictions, actual):
    if pred == 1:
        if act == 1:
            tp += 1
        else:
            fp += 1
    else:
        if act == 1:
            fn += 1
        else:
            tn += 1

precision = tp / (tp + fp)
recall = tp / (tp + fn)

Fscore = 2 * precision * recall / (precision + recall)

print("true positive:", tp, "\ntrue negative:", tn, "\nfalse negative:", fn, "\nfalse positive:", fp, "\nprecision:",
      precision, "\nrecall:", recall, "\nF1 score:", Fscore)