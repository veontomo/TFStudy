from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import random
import math 
import numpy as np


# http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

# Prepare the data

L = 6                    # dimension of the input vector
M = int(math.pow(3, L))  # the number of available input data (training data + cross-check data + verification data)
N = 6*M // 10            # size of the training set

print("Parameters:\nL = ", L, " - input vector dimension\nM = ", M, " - size of data set\nN = ", N, " - training data size")

# Create a list of integers from 1 to M-1 in a random order
inputIntegers = list(range(M))
random.shuffle(inputIntegers)

def base(n):
	"""Base-3 representation of number n"""
	b = 2 # define the base
	r = n // b
	if r == 0:
		return [n]
	else: 
		rest = base(r)
		rest.append(n % b)
		return rest


def padding(l):
	"""Pads given list to have a size L"""
	return (L*[0] + l)[-L:]

def getLabel(n):
	"""1, if n is even, 0 otherwise"""
	if (n % 2 == 0):
	    return 1
	else:
		return 0


dataBase = list(map(base, inputIntegers))
dataX = list(map(padding, dataBase))
dataY = list(map(getLabel, inputIntegers))

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
              metrics=['accuracy'])

model.fit(dataX, dataY, nb_epoch=20, batch_size=32)

for layer in model.layers:
	print(layer.get_weights())


predictions = list(map(lambda x: 1 if (x>0.5) else 0, model.predict(dataX[N:])))

for i in range(N, M):
	print(inputIntegers[i], dataY[i], predictions[i-N])

