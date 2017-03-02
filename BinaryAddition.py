import os
from keras.models import Sequential
from keras.layers import Dense, GRU, Activation, LSTM, TimeDistributed
#from keras.optimizers import SGD
#import random
#import math 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# from keras.utils.visualize_util import plot

dirName = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")

os.system('cls')

LOW = 0
HIGH = 127
L = 8
E = 100
NUM_TRAIN_INPUTS = 100
NUM_TOTAL_INPUTS = 128
raw = list(range(LOW, HIGH + 1))
s = len(raw)
X1 = []
X2 = []
for i in range(0, NUM_TOTAL_INPUTS):
	X1.append(raw[i * 7 % s ])
	X2.append(raw[i * 11 % s ])

# X1 = list(range(LOW, HIGH + 1))
# random.shuffle(X1)
# X2 = list(range(LOW, HIGH + 1))
# random.shuffle(X2)
Y = [x1 + x2 for x1, x2 in zip(X1, X2)]


def base(n, b):
	"""Base-b representation of number n"""
	r = n // b
	if r == 0:
		return [n]
	else: 
		rest = base(r, b)
		rest.append(n % b)
		return rest

def padding(l):
	"""Pads given list to have a size L"""
	return (L*[0] + l)[-L:]

X1base = list(map(lambda i: padding(base(i, 2)), X1))
X2base = list(map(lambda i: padding(base(i, 2)), X2))
Ybase = list(map(lambda i: padding(base(i, 2)), Y))

X = []
for (x1, x2) in zip(X1base, X2base):
	for i in range(0, len(x1)):
		X.append([x1[i], x2[i]]) 

X = np.array(X).reshape((-1, L, 2))
Y = np.array(Ybase).reshape((-1, L, 1))


print('Build model...')
model = Sequential()
model.add(LSTM(output_dim = 2*L, input_length = L, input_dim = 2, return_sequences = True, activation = 'linear', name="analyzer"))
model.add(Dense(L, activation='linear'))
model.add(TimeDistributed(Dense(2, activation='sigmoid')))
model.add(Dense(1, activation='sigmoid'))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['fbeta_score']) 
model.summary()
# plot(model, to_file= dirName + '/model_diagram.png')
history = model.fit(X[:NUM_TRAIN_INPUTS], Y[:NUM_TRAIN_INPUTS], nb_epoch=E, batch_size=8, verbose= 0)



# print('prediction', X[NUM_TRAIN_INPUTS:], model.predict(X[NUM_TRAIN_INPUTS:]))

for layer in model.layers:
	print(layer.name, 'input shape', layer.input_shape, 'output shape', layer.output_shape)
	print(layer.get_config())

plt.plot(list(range(1, E+1)), history.history['fbeta_score'], 'k', color='green')
plt.plot(list(range(1, E+1)), history.history['loss'], 'k', color='blue')
plt.xlabel('Epoch')
plt.title('Binary addition')

loss_line = mlines.Line2D([], [], color='blue', label='loss')
fscore_line = mlines.Line2D([], [], color='green', label='F1 score')
plt.legend(handles=[loss_line, fscore_line])

plt.show()
