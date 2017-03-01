import os
from keras.models import Sequential
from keras.layers import Dense, GRU, Activation, LSTM, TimeDistributed
from keras.optimizers import SGD
import random
import math 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

os.system('cls')
Xraw = [[0, 0], [0, 1], [1, 1], [0, 1], [1, 0], [1, 0], [1, 1], [1, 0]]
Yraw = [1, 0, 1, 0, 0, 0, 0, 1]

X = np.array(Xraw).reshape((-1,8,2))
Y = np.array(Yraw).reshape((1,8,1))

print('Xraw', Xraw)
print('X', X)
print('Yraw', Yraw)
print('Y', Y)

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(GRU(output_dim = 2, input_length = 8, input_dim = 2, return_sequences=True))
model.add(Activation('relu'))
model.add(TimeDistributed(Dense(2, activation='softmax')))
model.add(TimeDistributed(Dense(1, activation='softmax')))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy']) 
model.fit(X, Y, nb_epoch=10, batch_size=8, verbose= 0)


X2 = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [1, 0], [1, 0], [0, 0], [1, 0]]).reshape((1,8,2))
print(model.predict(X2))
model.summary()