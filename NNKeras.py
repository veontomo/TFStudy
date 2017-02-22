from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import random
import math 
import numpy as np


# Prepare the data

L = 3                    # dimension of the input vector
M = int(math.pow(3, L))  # the number of available input data (training data + cross-check data + verification data)
N = 8*M // 10            # size of the training set

print("Parameters:\nL = ", L, " - input vector dimension\nM = ", M, " - size of data set\nN = ", N, " - training data size")

# Create a list of integers from 1 to M-1 in a random order
inputIntegers = list(range(M))
random.shuffle(inputIntegers)

def base(n, b):
	"""Base-b representation of number n"""
	r = n // b
	if r == 0:
		return [n]
	else: 
		rest = base(r, b)
		rest.append(n % b)
		return rest

def base3(n):
	return base(n, 3)

def padding(l):
	"""Pads given list to have a size L"""
	return (L*[0] + l)[-L:]

def isEven(n):
	"""1, if n is even, 0 otherwise"""
	if (n % 2 == 0):
	    return 1
	else:
		return 0


dataBase3 = list(map(base3, inputIntegers))
dataX = list(map(padding, dataBase3))
dataY = list(map(isEven, inputIntegers))

print("X", dataX)
print("Y", dataY)

# extract the first elements in order to use tham as a trainig set
X_train = dataX[:N]
Y_train = dataY[:N]


data = np.random.random((1000, 784))
labels = np.random.randint(2, size=(1000, 1))

model = Sequential()
model.add(Dense(1, input_dim=784, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# train the model, iterating on the data in batches
# of 32 samples
model.fit(data, labels, nb_epoch=10, batch_size=32)

exit()
# https://keras.io/getting-started/sequential-model-guide/

model = Sequential()
model.add(Dense(1, input_dim=L, activation='sigmoid'))
# model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)
classes = model.predict_classes(X_test, batch_size=32)
proba = model.predict_proba(X_test, batch_size=32)