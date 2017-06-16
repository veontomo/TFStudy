from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

# Prepare the data

L = 6  # dimension of the input vector
M = int(math.pow(3, L))  # the number of available input data (training data + cross-check data + verification data)
N = 6 * M // 10  # size of the training set
E = 50  # the number of epochs
print("Parameters:\nL = ", L, " - input vector dimension\nM = ", M, " - size of data set\nN = ", N,
      " - training data size")

# Create a list of integers from 1 to M-1 in a random order
inputIntegers = list(range(M))
random.shuffle(inputIntegers)



def base(n):
    """Base-3 representation of number n"""
    b = 2  # define the base
    r = n // b
    if r == 0:
        return [n]
    else:
        rest = base(r)
        rest.append(n % b)
        return rest


def padding(l):
    """Pads given list to have a size L"""
    return (L * [0] + l)[-L:]


def getLabel(n):
    """1, if n is even, 0 otherwise"""
    if n % 2 == 0:
        return 1
    else:
        return 0


dataBase = list(map(base, inputIntegers))
dataX = list(map(padding, dataBase))
dataY = list(map(getLabel, inputIntegers))
print(dataX)
print(dataY)

# for i in range(N):
# print(inputIntegers[i], " X =", dataX[i], " Y =", dataY[i])

# extract the first elements in order to use tham as a trainig set
X_train = dataX[:N]
Y_train = dataY[:N]

model = Sequential()
model.add(Dense(L, input_dim=L, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
# model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['fbeta_score'])

history = model.fit(X_train, Y_train, nb_epoch=E, batch_size=32, verbose=0)

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

# visualize the training progress
plt.plot(list(range(1, E + 1)), history.history['fbeta_score'], 'k', color='green')
plt.plot(list(range(1, E + 1)), history.history['loss'], 'k', color='blue')
plt.xlabel('Epoch')
plt.title('Training progress')

loss_line = mlines.Line2D([], [], color='blue', label='loss')
fscore_line = mlines.Line2D([], [], color='green', label='F1 score')
plt.legend(handles=[loss_line, fscore_line])

plt.show()
