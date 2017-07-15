import numpy as np

import matplotlib.pyplot as plt
from keras.constraints import maxnorm
from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers import Dense, Activation, Dropout, Conv2D, Flatten
import matplotlib.lines as mlines
import math
from keras.utils import np_utils

# import keras

# print(keras.__version__)
# exit()
E = 100  # number of epochs
FRACTION = 0.8  # fraction of initial data to be used for cross-validation
LINE_NUMBERS = 200  # number of lines to read from 'train.csv'


def mod(x):
    return math.fabs(x)


def oneMinus(x):
    return 1 - x


def cross_binary_single(x):
    return math.log(1 - math.fabs(x))


crossBinaryVect = np.vectorize(cross_binary_single)


def loss(x, y):
    return -np.sum(crossBinaryVect(x - y)) / (x.shape[0] * x.shape[1])


with open("train.csv", encoding="ascii") as file:
    lst = [next(file) for x in range(0, LINE_NUMBERS)]

title = [v.strip() for v in lst[0].split(",")]
data = [[int(i) for i in v.strip().split(",")] for v in lst[1:]]

X = np.array(data)[:, 1:]
Y = np.array(data)[:, 0]

Ycateg = np_utils.to_categorical(Y)
Ximg = np.array([[np.reshape(x, [28, 28])] for x in X])
print(Ximg.shape)

F = X.shape[1]  # dim of the input vector (number of features)
M = Ycateg.shape[1]  # dim of the output vector
T = int(FRACTION * X.shape[0])  # number of train input vectors


def binary_crossentropy(X, Y):
    num = X.shape[0] * X.shape[1]
    return -sum([sum([math.log(1 - math.fabs(x)) for x in z]) for z in X - Y]) / num


lossAccum = []
weightHistory = []
bias0 = []


# class TestCallback(Callback):
#     def __init__(self, test_data):
#         self.test_data = test_data
#
#     def on_epoch_end(self, epoch, logs={}):
#         x, y = self.test_data
#         weightHistory.append(self.model.layers[0].get_weights()[0][0])
#         bias0.append(self.model.layers[0].get_weights()[0][1])
#         predicted = self.model.predict(x, verbose=0)
#         l = loss(y, predicted)
#         print('Epoch', epoch + 1, 'loss', l)
#         lossAccum.append(l)


X_train = Ximg[:T]
Y_train = Ycateg[:T]
X_cv = Ximg[T:]
Y_cv = Ycateg[T:]

print("number of features", F)
print("output dimension", M)
print("number of test data", T)
print("number of cross validation data", len(X_cv))
print("number of epochs", E)

model = Sequential()
model.add(Conv2D(28, (4, 4), input_shape=(1, 28, 28), padding='same', activation='relu', kernel_constraint=maxnorm(3)))

model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# history = model.fit(X_train, Y_train, nb_epoch=E, verbose=0)
history = model.fit(X_train, Y_train, epochs=E, verbose=0)
# history = model.fit(X_train, Y_train, nb_epoch=E, verbose=0, callbacks=[TestCallback((X_cv, Y_cv))])
# for layer in model.layers:
#    print(layer.get_weights())
wrongPred = {}
for h in range(0, len(X_cv)):
    pickElem = X_cv[[h]]
    predict = model.predict(pickElem)
    digitPred = np.argmax(predict)
    digitAct = np.argmax(Y_cv[[h]])
    if digitAct != digitPred:
        wrongPred[h] = [digitAct, digitPred]
# print('confused', digitAct, 'with', digitPred)

wrongPredSize = len(wrongPred)
plotRows = int(math.ceil(math.sqrt(wrongPredSize)))
if plotRows * plotRows < wrongPredSize:
    plotRows = plotRows + 1

# counter = 0
# for h in wrongPred:
#     plt.subplot(plotRows+1, plotRows, counter + 1)
#     plt.title(str(wrongPred[h][1]) + ' ' + str(wrongPred[h][0]))
#     plt.imshow(np.reshape(X_cv[[h]], [28, 28]), cmap='gray')
#     plt.axis('off')
#     counter = counter + 1
# print('Total amount of erroneus labels:', counter)
# plt.subplots_adjust(top=0.9, bottom=0.1, left=0.10, right=0.95, hspace=0.8, wspace=0.3)


COLOR_TRAIN = 'blue'
COLOR_CV = 'green'
#plt.subplot(plotRows + 1, 1, plotRows + 1)
plt.plot(range(1, E + 1), history.history['loss'], 'k', color=COLOR_TRAIN)
#plt.plot(range(1, E + 1), lossAccum, 'k', color=COLOR_CV)
plt.xlabel('Epoch')
plt.title('Training progress')

lineLegend = []
lineLegend.append(mlines.Line2D([], [], color=COLOR_TRAIN, label='train loss'))
lineLegend.append(mlines.Line2D([], [], color=COLOR_CV, label='cross validation loss'))
plt.legend(handles=lineLegend)

plt.show()

layerNum = 1
for layer in model.layers:
    weights = layer.get_weights()
    weightNum = 1
    for w in weights:
        print('layer n. ', layerNum, 'weight matrix n.', weightNum)
        print(w.shape)
        weightNum = weightNum + 1
        layerNum = layerNum + 1

# show input weight evolution
# counter = 1
# for w in weightHistory:
#     plt.subplot(5, (E // 5) + 1, counter)
#     plt.imshow(np.reshape(w, [28, 28]), cmap='gray')
#     plt.axis('off')
#     counter = counter + 1
# plt.show()

# show input bias evolution
counter = 1
historyCounter = 0
maxPlotNum = 40
if E > maxPlotNum:
    historyStep = E // maxPlotNum
else:
    historyStep = 1
for w in bias0[::historyStep]:
    plt.subplot((maxPlotNum // 5) + 2, 5, counter)
    plt.imshow(np.reshape(w, [28, 28]), cmap='gray')
    plt.axis('off')
    counter = counter + 1

plt.show()
