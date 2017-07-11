import numpy as np

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers import Dense, Activation, Dropout
import matplotlib.lines as mlines
import keras.objectives as losses

with open("train.csv", encoding="ascii") as file:
    lst = [next(file) for x in range(0, 100)]

title = [v.strip() for v in lst[0].split(",")]
data = [[int(i) for i in v.strip().split(",")] for v in lst[1:]]

X = np.array(data)[:, 1:]
Y = np.array(data)[:, 0]

Ysoft = np.array([[1 if (i == y) else 0 for i in range(0, 10)] for y in Y])

F = X.shape[1] # dim of the input vector (number of features)
M = Ysoft.shape[1] # dim of the output vector
T = int(0.8 * X.shape[0]) # number of train input vectors
E = 20

#print(Ysoft)
#plt.imshow(np.reshape(X[1], [28, 28]), cmap='gray')
#plt.show()

lossAccum = []
class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        predicted = self.model.predict(x, verbose=0)
        l = losses.binary_crossentropy(y, predicted)
        lossAccum.append(l)

X_train = X[:T]
Y_train = Ysoft[:T]
X_cv = X[T:]
Y_cv = Ysoft[T:]

print("number of features", F)
print("output dimension", M)
print("number of test data", T)
print("number of cross validation data", len(X_cv))
print("number of epochs", E)


model = Sequential()
model.add(Dense(F, input_dim=F, activation='tanh'))
model.add(Dense(F, activation='softmax'))
model.add(Dense(M, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#history = model.fit(X_train, Y_train, nb_epoch=E, verbose=0)
history = model.fit(X_train, Y_train, nb_epoch=E, verbose=0, callbacks=[TestCallback((X_cv, Y_cv))])
# for layer in model.layers:
#    print(layer.get_weights())

# visualize the training progress

COLOR_TRAIN = 'blue'
COLOR_CV = 'green'
#print(history.history.keys())
plt.plot(range(1, E + 1), history.history['loss'], 'k', color=COLOR_TRAIN)
plt.plot(range(1, E + 1), lossAccum, 'k', color=COLOR_CV)
plt.xlabel('Epoch')
plt.title('Training progress')

lineLegend = []
lineLegend.append(mlines.Line2D([], [], color=COLOR_TRAIN, label='train loss'))
lineLegend.append(mlines.Line2D([], [], color=COLOR_CV, label='cross validation loss'))
plt.legend(handles=lineLegend)

plt.show()
