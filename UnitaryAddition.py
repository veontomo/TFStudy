# Count how many times a binary input contains 1
# 
# See
# 
# http://peterroelants.github.io/posts/rnn_implementation_part01/
# 
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

def base(n, b):
	"""Base-b representation of number n"""
	r = n // b
	if r == 0:
		return [n]
	else: 
		rest = base(r, b)
		rest.append(n % b)
		return rest

def padding(l, s):
	"""Pads given list to have a size L"""
	return (s*[0] + l)[-s:]

dirName = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")
os.system('cls')

DATASET_SIZE = 64
INPUT_LEN = 6
TRAINING_SIZE = 2 * DATASET_SIZE // 3
E = 1000

X = np.array(list(map(lambda i: padding(base(i, 2), INPUT_LEN), range(0, DATASET_SIZE)))).reshape((-1, 1, INPUT_LEN))
Y = np.array(list(map(lambda x: sum(x[0]), X))).reshape((-1, 1, 1))
training = X[:TRAINING_SIZE]
labels= Y[:TRAINING_SIZE]

print('Build model...')
model = Sequential()
model.add(LSTM(output_dim = 1, input_length = 1, input_dim = INPUT_LEN, return_sequences = True, activation = 'linear', name="analyzer"))
model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['fbeta_score']) 
model.summary()
history = model.fit(training, labels, nb_epoch=E, batch_size=8, verbose= 0)

Xverification = X[TRAINING_SIZE:]
Yverification = Y[TRAINING_SIZE:]

prediction = model.predict(Xverification)
for (x, y, p) in zip(Xverification, Yverification, prediction):
	print(x, y, p)

for layer in model.layers:
	print(layer.name, 'input shape', layer.input_shape, 'output shape', layer.output_shape)
	print(layer.get_config())
	print(layer.get_weights())

plt.plot(list(range(1, E+1)), history.history['fbeta_score'], 'k', color='green')
plt.plot(list(range(1, E+1)), history.history['loss'], 'k', color='blue')
plt.xlabel('Epoch')
plt.title('Bit sum')

loss_line = mlines.Line2D([], [], color='blue', label='loss')
fscore_line = mlines.Line2D([], [], color='green', label='F1 score')
plt.legend(handles=[loss_line, fscore_line])

plt.show()
