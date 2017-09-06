# Next word prediction
# See http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
import os
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import sys

SEQUENCE_LENGTH = 20
NUM_UNITS = 512

# load ascii text and covert to lowercase
dirName = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")
filename = dirName + "/poems.txt"
fileContent = open(filename).read()
lines =[line.lower() for line in fileContent.split("\n") if not(line.startswith("#"))]
content = '\n'.join(lines)

# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(content)))
charToInt = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
# summarize the loaded data
charNum = len(content)
symbolNum = len(chars)
print("Total characters: ", charNum)
print("Total symbols: ", symbolNum)

# prepare the dataset of input to output pairs encoded as integers
dataX = []
dataY = []
for i in range(0, charNum - SEQUENCE_LENGTH, 1):
    seqIn = content[i:i + SEQUENCE_LENGTH]
    seqOut = content[i + SEQUENCE_LENGTH]
    dataX.append([charToInt[char] for char in seqIn])
    dataY.append(charToInt[seqOut])
patternNum = len(dataX)
print("Total Patterns: ", patternNum)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (patternNum, SEQUENCE_LENGTH, 1))
# normalize
X = X / float(symbolNum)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
print("number of units: ", NUM_UNITS)
print("X shape:", X.shape)
print("y shape:", y.shape)


# define the checkpoint
filepath="weights/wordpredict-{epoch:05d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model = Sequential()
model.add(LSTM(NUM_UNITS, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.summary()
# load the network weights
filename = dirName + "/weights/wordpredict-00105-0.0210.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
#model.fit(X, y, epochs=200, batch_size=32, callbacks=callbacks_list, initial_epoch = 0)
# pick a random seed
#start = numpy.random.randint(0, len(dataX) - 1)
#pattern = dataX[start]
phrase = "soft kitty, warm kitty little ball of fur"
pattern = [charToInt[c] for c in phrase[:SEQUENCE_LENGTH]]
print(pattern)
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
# generate characters
for i in range(100):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(symbolNum)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seqIn = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print("\nDone.")


