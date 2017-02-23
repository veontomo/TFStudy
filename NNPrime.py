import os
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import random
import math 
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.lines as mlines

dirName = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")
filename = dirName + '/primes_20000'
file = open(filename, 'r')
primes = list(map(lambda str: int(str.replace('\n', '')), file.readlines()[:100]))
file.close()
print(primes[:20])

maxNum = primes[-1]
print(maxNum)

randomIntegers = list(range(2, maxNum))
random.shuffle(randomIntegers)

L = 6                    # dimension of the input vector
M = len(primes)   # the number of available input data (training data + cross-check data + verification data)
N = 6*M // 10            # size of the training set
E = 50   


def base(n):
	"""Base-3 representation of number n"""
	b = 3 # define the base
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

input = list(map(padding, map(base, randomIntegers)))
labels = list(map(lambda x: 1 if (x in primes) else 0, randomIntegers))

for (i, x, y) in zip(randomIntegers, input, labels):
	print("i:", i, "x:", x, "y:", y)



# extract the first elements in order to use tham as a trainig set
X_train = input[:N]
Y_train = labels[:N]


model = Sequential()
model.add(Dense(L, input_dim=L, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
# model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['fbeta_score'])

history = model.fit(input, labels, nb_epoch=E, batch_size=32, verbose= 0)

for layer in model.layers:
	print(layer.get_weights())


predictions = list(map(lambda x: 1 if (x > 0.5) else 0, model.predict(input[N:])))
actual = labels[N:]

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

precision = tp / (tp + fp)
recall = tp / (tp + fn)

Fscore = 2*precision*recall/(precision + recall)

print("true positive:", tp, "\ntrue negative:", tn, "\nfalse negative:", fn, "\nfalse positive:", fp, "\nprecision:", precision, "\nrecall:", recall, "\nF1 score:", Fscore)
exit()
# visualize the training progress
plt.plot(list(range(1, E+1)), history.history['fbeta_score'], 'k', color='green')
plt.plot(list(range(1, E+1)), history.history['loss'], 'k', color='blue')
plt.xlabel('Epoch')
plt.title('Training progress')

loss_line = mlines.Line2D([], [], color='blue', label='loss')
fscore_line = mlines.Line2D([], [], color='green', label='F1 score')
plt.legend(handles=[loss_line, fscore_line])

plt.show()
