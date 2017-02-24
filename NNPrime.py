import os
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import SGD
import random
import math 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


# Generqte input data
# The input data is a list of integers with labels 0 (if the number is not a prime one) or 1 (if the number is a prime one).

dirName = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")
filename = dirName + '/primes_20000'
file = open(filename, 'r')
primes = list(map(lambda str: int(str.replace('\n', '')), file.readlines()[:5000]))
file.close()

isPrime = []

minNum = primes[0]
maxNum = primes[-1]
size = len(primes)

isPrime = []

for i in range(0, size - 1):
	start = primes[i]
	end = primes[i+1]
	isPrime.append(1)
	quantity = end - start - 1
	if (quantity > 0):
		isPrime.extend([0] * quantity)
isPrime.append(1)

shuffledIntegers = list(range(minNum, maxNum+1))
random.shuffle(shuffledIntegers)
dataY = list(map(lambda x: isPrime[x - minNum], shuffledIntegers))


L = 6                    # dimension of the input vector
M = len(shuffledIntegers)           # the number of available input data (training data + cross-check data + verification data)
N = 6*M // 10            # size of the training set
E = 20   


def base(n):
	"""Base-10 representation of number n"""
	b = 10 # define the base
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

dataX = list(map(padding, map(base, shuffledIntegers)))




# extract the first elements in order to use tham as a trainig set
X_train = dataX[:N]
Y_train = dataY[:N]

model = Sequential()
model.add(TimeDistributed(Dense(8, input_dim=L, Activation='softmax')))
model.add(LSTM(4, dropout_W=0.2, dropout_U=0.2))
model.add(Dense(1))
# model.add(Dense(2, input_dim=L, activation='tanh'))
# model.add(SimpleRNN(1))
# model.add(Dense(4, activation='sigmoid'))
# model.add(Dense(1, activation='sigmoid'))
# model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['fbeta_score'])

history = model.fit(X_train, Y_train, nb_epoch=E, batch_size=32, verbose= 0)

for layer in model.layers:
	print(layer.get_weights())

predictedValues = model.predict(dataX[N:])
predictions = list(map(lambda x: 1 if (x > 0.5) else 0, predictedValues))
actual = dataY[N:]

# for (v, p, a, n) in zip(predictedValues, predictions, actual, dataX[N:]):
	# print(n, v, p, a)

# Calculate F score on the cross-validation data
tp = 0
tn = 0
fp = 0
fn = 0
for (pred, act) in zip(predictions, actual):
	if pred == 1: 
		if act == 1:
			tp = tp + 1
		else: 
			fp = fp + 1
	else: 
		if act == 1:
			fn = fn + 1
		else:
			tn = tn + 1
print("true positive:", tp, "\ntrue negative:", tn, "\nfalse negative:", fn, "\nfalse positive:", fp)

precision = tp / (tp + fp) if (tp + fp != 0) else 0
recall = tp / (tp + fn) if (tp + fn != 0) else 0

Fscore = 2*precision*recall/(precision + recall) if (precision + recall != 0) else 0

print("precision:", precision, "\nrecall:", recall, "\nF1 score:", Fscore)

# visualize the training progress
plt.plot(list(range(1, E+1)), history.history['fbeta_score'], 'k', color='green')
plt.plot(list(range(1, E+1)), history.history['loss'], 'k', color='blue')
plt.xlabel('Epoch')
plt.title('Prime number detection')

loss_line = mlines.Line2D([], [], color='blue', label='loss')
fscore_line = mlines.Line2D([], [], color='green', label='F1 score')
plt.legend(handles=[loss_line, fscore_line])

plt.show()
