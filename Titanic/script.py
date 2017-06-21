#import os
#import tensorflow as tf
import re
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from sklearn import ensemble

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


def extract_title(str):
    pattern = ".*?,\s+(.*?)\."
    if re.match(pattern, str):
        res = re.search(pattern, str)
        return res.group(1)
    print("Title not found in", str)
    return ""

# Read the data into a list
with open("train.csv", encoding="ascii") as file:
    list = file.readlines()

# First element of the list contains the feature names
orig_features = list[0].strip().split(",")
constructed_features = orig_features[:]
constructed_features.append("Deck")
constructed_features.append("CabinNumber")
constructed_features.append("Title")

# features that should be treated as integers
intFeatures = ["PassengerId", "Survived", "Pclass", "SibSp", "Parch"]
# features that should be treated as floats
floatFeatures = ["Age", "Fare"]

feature_values = {key: [] for key in constructed_features}

missing_features = {key: 0 for key in orig_features}

# iterate over the data set in order to collect all possible feature values
for i in range(1, len(list)):
    line = re.split(",(?!\s)", list[i].strip())
    for k in range(0, len(line)):
        featureName = orig_features[k]
        value = line[k]
        if value == "":
            missing_features[orig_features[k]] += 1
        if featureName == "Cabin":
            cabin_pair = cabin(value)
            if not(cabin_pair[0] in feature_values["Deck"]):
                feature_values["Deck"].append(cabin_pair[0])
                feature_values["Deck"].sort()
            if not(cabin_pair[1] in feature_values["CabinNumber"]):
                feature_values["CabinNumber"].append(cabin_pair[1])
                feature_values["CabinNumber"].sort()
        if featureName == "Name":
            title = extract_title(value)
            if not(title in feature_values["Title"]):
                feature_values["Title"].append(title)
        if featureName in intFeatures:
            cast_value = int(value)
        elif featureName in floatFeatures:
            if value != '':
                cast_value = float(value)
            else:
                cast_value = -1
        else:
            cast_value = value
        if not(cast_value in feature_values[featureName]):
            feature_values[featureName].append(cast_value)
            feature_values[featureName].sort()

print("missing features", missing_features)

# indexed version of the data set (in order to avoid strings)
digitalized = []

for i in range(1, len(list)):
    tmpLine = []
    tmpCabin = []
    titleIndex = -1
    line = re.split(",(?!\s)", list[i].strip())
    for k in range(0, len(line)):
        if orig_features[k] in intFeatures:
            cast_value = int(line[k])
        elif orig_features[k] in floatFeatures:
            if line[k] != '':
                cast_value = float(line[k])
            else:
                cast_value = -1
        else:
            cast_value = line[k]
        tmpLine.append(feature_values[orig_features[k]].index(cast_value))
        if orig_features[k] == "Cabin":
            cabin_pair = cabin(line[k])
            tmpCabin.append(feature_values["Deck"].index(cabin_pair[0]))
            tmpCabin.append(feature_values["CabinNumber"].index(cabin_pair[1]))
        if orig_features[k] == "Name":
            title = extract_title(line[k])
            titleIndex = feature_values["Title"].index(title)
    if titleIndex < 0:
        print("in line n.", i, "title", title, "is not found")
    tmpLine.append(tmpCabin[0])
    tmpLine.append(tmpCabin[1])
    tmpLine.append(titleIndex)
    digitalized.append(tmpLine)

pos = [constructed_features.index(key) for key in
       ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Deck", "CabinNumber", "Title"]]
# input data
dataX = np.array(digitalized)[:, pos]
# labels
dataY = np.array(digitalized)[:, constructed_features.index("Survived")]

L = len(pos)
N = int(0.8 * len(dataY))
E = 200

print("number of features", L)
print("number of training data", N)
print("number of epochs", E)

X_train = dataX[:N]
Y_train = dataY[:N]

X_test = dataX[N:]
Y_test = dataY[N:]

#
# http://slides.com/simonescardapane/machine-learning-from-a-developer-s-pov#/7
rf = ensemble.RandomForestClassifier().fit(X_train, Y_train)
print('Predicted league is:', rf.predict(X_test))

exit()
model = Sequential()
model.add(Dense(L, input_dim=L, activation='linear', init='normal'))
model.add(Dense(1, activation='sigmoid', init='normal'))
#model.add(Dense(1, init='normal', activation='relu'))
model.compile(optimizer='RMSprop',
              loss='binary_crossentropy',
              metrics=['fbeta_score'])


#model.add(Dense(60, input_dim=60, init='normal', activation='relu'))
#model.add(Dense(1, init='normal', activation='sigmoid'))
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

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
print(tp, "+", tn, "+", fn, "+", fp, "=" if tp + fp + fn + tn == len(X_test) else "<>", len(X_test))

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