#import os
#import tensorflow as tf
import re
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def cabin(str):
    """Split the cabin number into a string and an integer"""
    if str == "":
        return ["", 0]
    regex1 = "([A-G\s]+)(\d+)"
    if re.match(regex1, str):
        parts = re.search(regex1, str)
        part1 = parts.group(1)
        part2 = parts.group(2)
        return [part1,  int(part2)]
    regex2 = "([A-GT\s]+)"
    if re.match(regex2, str):
        parts = re.search(regex2, str)
        part1 = parts.group(1)
        return [part1, 0]
    regex3 = "(\d+)"
    if re.match(regex3, str):
        parts = re.search(regex3, str)
        part1 = parts.group(1)
        return ["", int(part1)]
    return ["", 0]

# Read the data into a list
with open("train.csv", encoding="ascii") as file:
    list = file.readlines()

# First element of the list contains the feature names
orig_features = list[0].strip().split(",")
generated_features = orig_features[:]
generated_features.append("Deck")
generated_features.append("CabinNumber")

# features that should be treated as integers
intFeatures = ["PassengerId", "Survived", "Pclass", "SibSp", "Parch"]
# features that should be treated as floats
floatFeatures = ["Age", "Fare"]

feature_values = {key: [] for key in generated_features}

# iterate over the data set in order to collect all possible feature values
for i in range(1, len(list)):
    line = re.split(",(?!\s)", list[i].strip())
    for k in range(0, len(line)):
        featureName = orig_features[k]
        if featureName == "Cabin":
            values = cabin(line[k])
            if not(values[0] in feature_values["Deck"]):
                feature_values["Deck"].append(values[0])
                feature_values["Deck"].sort()
            if not(values[1] in feature_values["CabinNumber"]):
                feature_values["CabinNumber"].append(values[1])
                feature_values["CabinNumber"].sort()
        if featureName in intFeatures:
            value = int(line[k])
        elif featureName in floatFeatures:
            if line[k] != '':
                value = float(line[k])
            else:
                value = -1
        else:
            value = line[k]
        if not(value in feature_values[featureName]):
            feature_values[featureName].append(value)
            feature_values[featureName].sort()

# indexed version of the data set (in order to avoid strings)
digitalized = []

for i in range(1, len(list)):
    tmpLine = []
    tmpCabin = []
    line = re.split(",(?!\s)", list[i].strip())
    for k in range(0, len(line)):
        if orig_features[k] in intFeatures:
            value = int(line[k])
        elif orig_features[k] in floatFeatures:
            if line[k] != '':
                value = float(line[k])
            else:
                value = -1
        else:
            value = line[k]
        tmpLine.append(feature_values[orig_features[k]].index(value))
        if orig_features[k] == "Cabin":
            values = cabin(line[k])
            tmpCabin.append(feature_values["Deck"].index(values[0]))
            tmpCabin.append(feature_values["CabinNumber"].index(values[1]))
    tmpLine.append(tmpCabin[0])
    tmpLine.append(tmpCabin[1])
    digitalized.append(tmpLine)

#pos = [orig_features.index("Pclass"), orig_features.index("Sex"), orig_features.index("Age"), orig_features.index("SibSp"), orig_features.index("Parch"), orig_features.index("Fare"), orig_features.index("Cabin"), orig_features.index("Embarked")]
pos = [generated_features.index(key) for key in ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Deck", "CabinNumber"]]
# input data
dataX = np.array(digitalized)[:, pos]
# labels
dataY = np.array(digitalized)[:, generated_features.index("Survived")]

L = len(pos)
N = int(0.8 * len(dataY))
E = 500

X_train = dataX[:N]
Y_train = dataY[:N]


model = Sequential()
model.add(Dense(L, input_dim=L, activation='sigmoid'))
model.add(Dense(L, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['fbeta_score'])

history = model.fit(X_train, Y_train, nb_epoch=E, batch_size=32, verbose=0)

for layer in model.layers:
    print(layer.get_weights())

predictions = map(lambda x: 1 if (x > 0.5) else 0, model.predict(dataX[N:]))
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

print("true positive:", tp, "\ntrue negative:", tn, "\nfalse negative:", fn, "\nfalse positive:", fp)

denom1 = tp + fp
denom2 = tp + fn
if denom1!= 0:
    precision = tp / denom1
    print("precision:", precision)
else:
    precision = -1
if denom2 != 0:
    recall = tp / denom2
    print("recall:", recall)
else:
    recall = -1
if (recall != -1) and (precision != -1):
    Fscore = 2 * precision * recall / (precision + recall)
    print("F1 score:", Fscore)

# visualize the training progress
plt.plot(range(1, E + 1), history.history['fbeta_score'], 'k', color='green')
plt.plot(range(1, E + 1), history.history['loss'], 'k', color='blue')
plt.xlabel('Epoch')
plt.title('Training progress')

loss_line = mlines.Line2D([], [], color='blue', label='loss')
fscore_line = mlines.Line2D([], [], color='green', label='F1 score')
plt.legend(handles=[loss_line, fscore_line])

plt.show()