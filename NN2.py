import os
import tensorflow as tf
import numpy as np
import random
logDir = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/") + "/log/"
print("Write the log to folder: " + logDir)
print("tensorflow version: " + tf.__version__)
L = 4 # dimension of the input vector
M = 3*3*3*3 # the number of available input data (training data + cross-check data + verification data)
N = 4*M // 5  # size of the training set

# trX = [[0,2,0], [0,2,1]]
# trY = [1,0]
# Create a list of integers from 1 to M-1 in a random order
data = list(range(M))
random.shuffle(data)

def base(n, b):
	r = n // b
	"Base-b representation of number n"
	if r == 0:
		return [n]
	else: 
		rest = base(r, b)
		rest.append(n % b)
		return rest

def base3(n):
	return base(n, 3)

def padding(n, l):
	return (n*[0] + l)[-n:]

def padding4(l):
	return padding(4, l)

def isEven(n):
	"0, if n is even, 1 otherwise"
	return n % 2

data2 = list(map(base3, data))
dataX = list(map(padding4, data2))
dataY = list(map(isEven, data))

# extract the first elements in order to use tham as a trainig set
trainingX = dataX[:N]
trainingY = dataY[:N]

X = tf.placeholder("int8", name="inputX") # create symbolic variables
Y = tf.placeholder("int8", name="inputY")

def model(X, w):
	return tf.reduce_sum(tf.multiply(tf.cast(X, tf.float64), w)) # lr is just X*w so this model line is pretty simple

w = tf.Variable(np.random.rand(1, L), name='weight')
 
y_model = model(X, w)

cost = tf.square(tf.cast(Y, tf.float64) - y_model) # use square error for cost function

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost) # construct an optimizer to minimize cost and fit line to my data

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize variables (in this case just variable W)
    tf.global_variables_initializer().run()

    for i in range(1000):
        for (x, y) in zip(trainingX, trainingY):
            # print("x", x, "y", y, "w", w)
            sess.run(train_op, feed_dict={X: x, Y: y})

    print(sess.run(w))

    for i in range(N, M):
        print(data[i], dataX[i], dataY[i], sess.run(model(dataX[i], w)))

