import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
import matplotlib.lines as mlines
import math
from keras.utils import np_utils
import os

Filters = 2
kernelSize = [20, 30]

WIDTH = 2
HEIGHT = 2
CHANNELS = 1
N = 3
Xtmp = [[[[w + h + i + c for w in range(0, HEIGHT)] for h in range(0, WIDTH)] for c in range(0, CHANNELS)] for i in
        range(0, N)]
X = np.array(Xtmp)
Ytmp = np.array([i % 2 for i in range(0, N)])
Y = np_utils.to_categorical(Ytmp)
CLASSES = Y.shape[1]
print(X.shape)
print(X)
print(Y.shape)
print(Y)

model = Sequential()
model.add(
    Conv2D(Filters, (kernelSize[0], kernelSize[1]), input_shape=(CHANNELS, WIDTH, HEIGHT), use_bias=False,
           padding='same',
           activation='sigmoid'))
model.add(Flatten())
model.add(Dense(CLASSES, activation='softmax'))

print(model.summary())

# plot_model(model, to_file='model.png')
exit()
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(model.layers[0].output)
history = model.fit(X, Y, epochs=1)
# history = model.fit(X_train, Y_train, nb_epoch=E, verbose=0, callbacks=[TestCallback((X_cv, Y_cv))])
# for layer in model.layers:
#    print(layer.get_weights())


model2 = Sequential()
model2.add(Conv2D(Filters, (kernelSize[0], kernelSize[1]), input_shape=(1, 28, 28), use_bias=False, padding='same',
                  activation='sigmoid', weights=model.layers[0].get_weights()))
predictFirstLayer = model2.predict(X)
print('first layer:', X.shape, ' -> ', predictFirstLayer.shape)
exit()
