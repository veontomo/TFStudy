import os
import tensorflow as tf
import numpy as np
import random
logDir = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/") + "/log/"
print("Write the log to folder: " + logDir)
print("tensorflow version: " + tf.__version__)

L = 5         # dimension of the input vector
M = 3*3*3*3*3 # the number of available input data (training data + cross-check data + verification data)
N = 9*M // 10  # size of the training set

print("Parameters: ", L, " - input vector dimension", M, " - size of data set", N, " - training data size")

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

def appendBias(l):
	return [1] + l

def isEven(n):
	"""[1], if n is even, [0] otherwise"""
	return [1-(n % 2)]

dataBase3 = list(map(base3, inputIntegers))
dataX = list(map(appendBias, list(map(padding, dataBase3))))
dataY = list(map(isEven, inputIntegers))

# extract the first elements in order to use tham as a trainig set
trainingX = dataX[:N]
trainingY = dataY[:N]

X = tf.placeholder("int8", name="inputX") # create symbolic variables
Y = tf.placeholder("int8", name="inputY")


def Z(X, w):
	return tf.matmul(tf.cast(X, tf.float64), w, False, True)

def sigmoid(X, w):
	return tf.sigmoid(Z(X, w))

# create a weight vector with a bias term
w = tf.Variable(np.random.rand(1, L + 1), name='weight')
one = tf.constant(1.0, tf.float64)
# define the cost function
#cost = tf.square(tf.cast(Y, tf.float64) - model(X, w)) 
# define the sigmoid cost
#cost = tf.square(tf.cast(Y, tf.float64) - sigmoid(X, w)) 
# cost = - tf.multiply(tf.cast(Y, tf.float64), tf.log(sigmoid(X, w))) - tf.multiply(one - tf.cast(Y, tf.float64), tf.log(one - sigmoid(X, w)))
cost = - tf.reduce_sum(tf.multiply(tf.cast(Y, tf.float64), tf.log(sigmoid(X, w)))) - tf.reduce_sum(tf.multiply(tf.cast( Y, tf.float64), tf.log( sigmoid(X, w))))


# accumError = tf.reduce_mean(tf.square(tf.matmul(tf.cast(X, tf.float64), w, False, True) - tf.cast(tf.transpose(Y), tf.float64)))

# accumError = - tf.matmul(tf.cast(Y, tf.float64), tf.log(tf.sigmoid(tf.matmul(tf.cast(X, tf.float64), w, False, True))))	

# construct an optimizer to minimize cost and fit line to my data
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost) 

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize variables (in this case just variable W)
    tf.global_variables_initializer().run()

    # for i in range(10):
        # sess.run(train_op, feed_dict={X: trainingX, Y: trainingY})

    print(sess.run(w))
    print(sess.run(Y, feed_dict={Y: trainingY}))
    print(sess.run(sigmoid(X, w), feed_dict={X : trainingX}))
    print(sess.run(cost, feed_dict={X: trainingX, Y: trainingY}))
    print(sess.run(one.dims, feed_dict={X: trainingX, Y: trainingY}))
    
    # trainingError = sess.run(cost, feed_dict={X: trainingX, Y: trainingY})
    # crossError = sess.run(cost, feed_dict={X: dataX[N:], Y: dataY[N:]})
    # print("predictions")
    # for i in range(N, M):
        # print(inputIntegers[i], dataX[i], dataY[i], sess.run(sigmoid(dataX[i], w)))

    # print("on training set")
    # for i in range(min(N // 10, 3)):
        # print(inputIntegers[i], dataX[i], dataY[i], sess.run(sigmoid(dataX[i], w)))

    # print("training error: ", trainingError)
    # print("cross error: ", crossError)
