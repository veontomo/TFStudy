# http://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/
#

import numpy as np
import matplotlib.pyplot as plt
from keras.constraints import maxnorm
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Activation, Dropout, Conv2D, Flatten, MaxPooling2D, BatchNormalization
import matplotlib.lines as mlines
import math
from keras.utils import np_utils
import os

E = 30  # number of epochs
FRACTION_TRAIN = 0.7  # fraction of initial data to be used for training.
FRACTION_CV = 0.1  # fraction of initial data to be used for cross-validation.
FRACTION_TEST = 1 - FRACTION_CV - FRACTION_TRAIN  # fraction of initial data to be used for test

if FRACTION_TRAIN < 0:
    print("Fraction for the train data must be non-negative")
    exit()

if FRACTION_CV < 0:
    print("Fraction for the cross-validation data must be non-negative")
    exit()

if FRACTION_TEST < 0:
    print("Fraction for the test data must be non-negative")
    exit()

TOTAL_LINE_NUMBERS = 42001
LINE_NUMBERS = 30000  # number of lines to read from 'train.csv'

HEIGHT = 28
WIDTH = 28
CHANNELS = 1
dir = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/") + "/"


def mod(x):
    return math.fabs(x)


def oneMinus(x):
    return 1 - x


def cross_binary_single(x):
    return math.log(1 - math.fabs(x))


crossBinaryVect = np.vectorize(cross_binary_single)


def loss(x, y):
    return -np.sum(crossBinaryVect(x - y)) / (x.shape[0] * x.shape[1])


source = dir + "train.csv"

with open(source, encoding="ascii") as file:
    #    lst = file.readlines()
    lst = [next(file) for x in range(0, LINE_NUMBERS)]

title = [v.strip() for v in lst[0].split(",")]
data = [[int(i) for i in v.strip().split(",")] for v in lst[1:]]

X = np.array(data)[:, 1:] / 255.0 - 0.5
Y = np.array(data)[:, 0]
Ycateg = np_utils.to_categorical(Y)
Ximg = np.array([np.reshape(np.array(x), [HEIGHT, WIDTH, CHANNELS]) for x in X])

F = X.shape[1]  # dim of the input vector (number of features)
M = Ycateg.shape[1]  # dim of the output vector
TRAIN_QTY = int(FRACTION_TRAIN * X.shape[0])  # number of train input vectors
CV_QTY = int(FRACTION_CV * X.shape[0])


def binary_crossentropy(X, Y):
    num = X.shape[0] * X.shape[1]
    return -sum([sum([math.log(1 - math.fabs(x)) for x in z]) for z in X - Y]) / num


lossAccum = []
weightHistory = []

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


X_train = Ximg[:TRAIN_QTY]
Y_train = Ycateg[:TRAIN_QTY]
X_cv = Ximg[TRAIN_QTY:(TRAIN_QTY + CV_QTY)]
Y_cv = Ycateg[TRAIN_QTY:(TRAIN_QTY + CV_QTY)]
X_test = Ximg[(TRAIN_QTY + CV_QTY):]
Y_test = Ycateg[(TRAIN_QTY + CV_QTY):]

print("number of features", F)
print("output dimension", M)
print("number of test data", TRAIN_QTY)
print("number of cross validation data", len(X_cv))
print("number of epochs", E)

Filters1 = 50
Filters2 = 35

kernelSize1 = [5, 5]
kernelSize2 = [5, 5]
pool_1 = (4, 4)
pool_2 = (4, 4)

# use some realistic network
X_input = Input((28, 28, 1))
# CONV -> BN -> RELU Block applied to X
X = Conv2D(16, (7, 7), strides = (1, 1), padding='same', name = 'conv0')(X_input)
X = BatchNormalization(axis = 3, name = 'bn0')(X)
X = Activation('relu')(X)

# MAXPOOL
X = MaxPooling2D((2, 2), name='max_pool_1')(X)


# CONV -> BN -> RELU Block applied to X
X = Conv2D(32, (7, 7), strides = (1, 1), padding='same')(X)
X = BatchNormalization(axis = 3)(X)
X = Activation('relu')(X)

# MAXPOOL
X = MaxPooling2D((2, 2), name='max_pool_2')(X)



# CONV -> BN -> RELU Block applied to X
X = Conv2D(16, (3, 3), strides = (1, 1))(X)
X = BatchNormalization(axis = 3)(X)
X = Activation('relu')(X)

# MAXPOOL
X = MaxPooling2D((2, 2))(X)

# FLATTEN X (means convert it to a vector) + FULLYCONNECTED
X = Flatten()(X)
X = Dense(256, activation='sigmoid')(X)
X = Dense(32, activation='sigmoid')(X)
X = Dense(10, activation='softmax')(X)


# Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
model = Model(inputs = X_input, outputs = X, name='HappyModel')

#model = Input(shape=(28, 28))
#model = Conv2D(96, (11, 11), padding='same', activation='relu')(model)
#model = Flatten()(model)



#layer1 = Conv2D(Filters1, (kernelSize1[0], kernelSize1[1]),
#                input_shape=(HEIGHT, WIDTH, CHANNELS),
#                use_bias=False,
#                padding='same', activation='sigmoid')
#model.add(layer1)
#model.add(Dropout(0.2))
#model.add(MaxPooling2D(pool_size=pool_1, strides=None, padding='same', data_format=None))
#model.add(Conv2D(Filters2, kernelSize2, activation='sigmoid', padding='same', kernel_constraint=maxnorm(3)))
#model.add(BatchNormalization(axis=3))
#model.add(Dropout(0.2))
#model.add(MaxPooling2D(pool_size=pool_2, strides=None, padding='same', data_format=None))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Flatten())
#model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
#model.add(Dropout(0.5))
#model.add(Dense(M, activation='softmax'))

model.summary()

# plot_model(model, to_file='model.png')
# exit()
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(model.layers[0].output)
history = model.fit(X_train, Y_train, validation_data=(X_cv, Y_cv), epochs=E)
# history = model.fit(X_train, Y_train, nb_epoch=E, verbose=0, callbacks=[TestCallback((X_cv, Y_cv))])
# for layer in model.layers:
#    print(layer.get_weights())


#model2 = Sequential()
#model2.add(
#    Conv2D(Filters1, (kernelSize1[0], kernelSize1[1]),
#           input_shape=(HEIGHT, WIDTH, CHANNELS),
#           use_bias=False, padding='same',
#           activation='sigmoid',
#           kernel_constraint=maxnorm(3),
#           weights=model.layers[0].get_weights())
#)
#predictFirstLayer = model2.predict(X_cv)
#print('first layer:', X_cv.shape, ' -> ', predictFirstLayer.shape)

MAX_ROWS = 10
MAX_SIZE = MAX_ROWS*MAX_ROWS
wrongPred = {}
for h in range(0, len(X_cv)):
    pickElem = X_test[[h]]
    predict = model.predict(pickElem)
    digitPred = np.argmax(predict)
    digitAct = np.argmax(Y_cv[[h]])
    if digitAct != digitPred:
        wrongPred[h] = [digitAct, digitPred]
    if len(wrongPred) > MAX_SIZE:
        break
# print('confused', digitAct, 'with', digitPred)

wrongPredSize = len(wrongPred)
plotRows = int(math.ceil(math.sqrt(wrongPredSize)))
if plotRows * plotRows < wrongPredSize:
    plotRows = plotRows + 1

counter = 0
plt.figure(1)
for h in wrongPred:
    plt.subplot(plotRows + 1, plotRows, counter + 1)
    plt.title(str(wrongPred[h][1]) + ' ' + str(wrongPred[h][0]))
    plt.imshow(np.reshape(X_cv[[h]], [28, 28]), cmap='gray')
    plt.axis('off')
    counter = counter + 1
print('Total amount of erroneous labels:', counter, 'out of', len(X_cv))
plt.subplots_adjust(top=0.9, bottom=0.1, left=0.10, right=0.95, hspace=0.8, wspace=0.3)
plt.draw()

COLOR_TRAIN = 'blue'
COLOR_CV = 'green'
lineLegend = []
lineLegend.append(mlines.Line2D([], [], color=COLOR_TRAIN, label='train'))
lineLegend.append(mlines.Line2D([], [], color=COLOR_CV, label='cross validation'))

plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(range(1, E + 1), history.history['loss'], 'k', color=COLOR_TRAIN)
plt.plot(range(1, E + 1), history.history['val_loss'], 'k', color=COLOR_CV)
plt.title('Loss')
plt.legend(handles=lineLegend, loc=1)

plt.subplot(2, 1, 2)
plt.plot(range(1, E + 1), history.history['acc'], 'k', color=COLOR_TRAIN)
plt.plot(range(1, E + 1), history.history['val_acc'], 'k', color=COLOR_CV)
plt.xlabel('Epoch')
plt.title('Accuracy')
plt.legend(handles=lineLegend, loc=4)

# plt.show()

print('train data: ', X_train.shape)
for layer in range(0, len(model.layers)):
    weights = model.layers[layer].get_weights()
    wMax = len(weights)
    if wMax == 0:
        print('layer', layer, ' has no weights')
    for w in range(0, wMax):
        print('layer n. ', layer, 'weight matrix n.', w)
        print(weights[w].shape)

#inputLayerWeights = model.layers[0].get_weights()[0]
#print(inputLayerWeights.shape)

#plt.figure(3)
#plt.title('First layer weights (inverted)')
#for i in range(0, Filters1):
#    plt.subplot((Filters1 // 5) + 2, 5, i + 1)
#    matrix = 1 - np.array(inputLayerWeights[:, :, 0, i])
#    plt.imshow(np.reshape(matrix, kernelSize1), cmap='gray')
#    plt.axis('off')
plt.draw()
plt.show()
# show input weight evolution
# counter = 1
# for w in weightHistory:
#     plt.subplot(5, (E // 5) + 1, counter)
#     plt.imshow(np.reshape(w, [28, 28]), cmap='gray')
#     plt.axis('off')
#     counter = counter + 1
# plt.show()
