import os
import tensorflow as tf
import numpy as np
import random
logDir = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/") + "/log/"
print("Write the log to folder: " + logDir)
print("tensorflow version: " + tf.__version__)

L = 5         # dimension of the input vector
M = 3*3*3*3*3 # the number of available input data (training data + cross-check data + verification data)
N = 4*M // 5  # size of the training set

print("Parameters: ", L, " - input vector dimension", M, " - size of data set", N, " - training data size")

# Create a list of integers from 1 to M-1 in a random order
inputIntegers = list(range(M))
random.shuffle(inputIntegers)

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

def padding(l):
	"Pads given list to have a size L"
	return (L*[0] + l)[-L:]

def appendBias(l):
	return [1] + l

def isEven(n):
	"0, if n is even, 1 otherwise"
	return n % 2

dataBase3 = list(map(base3, inputIntegers))
dataX = list(map(appendBias, list(map(padding, dataBase3))))
dataY = list(map(isEven, inputIntegers))

# extract the first elements in order to use tham as a trainig set
trainingX = dataX[:N]
trainingY = dataY[:N]

X = tf.placeholder("int8", name="inputX") # create symbolic variables
Y = tf.placeholder("int8", name="inputY")

def model(X, w):
	return tf.reduce_sum(tf.multiply(tf.cast(X, tf.float64), w)) 

# create a weight vector with a bias term
w = tf.Variable(np.random.rand(1, L + 1), name='weight')

# define the cost function
cost = tf.square(tf.cast(Y, tf.float64) - model(X, w)) 

accumError = tf.reduce_sum(tf.square(tf.matmul(tf.cast(X, tf.float64), w, False, True) - tf.cast(tf.transpose(Y), tf.float64)))

# construct an optimizer to minimize cost and fit line to my data
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost) 

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize variables (in this case just variable W)
    tf.global_variables_initializer().run()

    for i in range(10):
        for (x, y) in zip(trainingX, trainingY):
            sess.run(train_op, feed_dict={X: x, Y: y})
    print(sess.run(w))
    accumError = sess.run(accumError, feed_dict={X: trainingX, Y: [trainingY]})
    print("accum error: ", accumError)
    for i in range(N, M):
        print(inputIntegers[i], dataX[i], dataY[i], sess.run(model(dataX[i], w)))

    print("on training set")
    for i in range(min(N // 10, 3)):
        print(inputIntegers[i], dataX[i], dataY[i], sess.run(model(dataX[i], w)))