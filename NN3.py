#!/usr/bin/env python

# https://github.com/nlintz/TensorFlow-Tutorials/blob/master/01_linear_regression.py
import tensorflow as tf
import numpy as np
import os

logDir = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/") + "/log/"

trX = np.linspace(-1, 1, 15)
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33 # create a y value which is approximately linear but with some random noise

print(trX)
print(trX)


X = tf.placeholder("float", name="inputX") # create symbolic variables
Y = tf.placeholder("float", name="inputY")


def model(X, w):
    return tf.multiply(X, w) # lr is just X*w so this model line is pretty simple


w = tf.Variable(0.0, name="weights") # create a shared variable (like theano.shared) for the weight matrix
y_model = model(X, w)

cost = tf.square(Y - y_model) # use square error for cost function

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost) # construct an optimizer to minimize cost and fit line to my data

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize variables (in this case just variable W)
    tf.global_variables_initializer().run()

    for i in range(100):
        for (x, y) in zip(trX, trY):
            sess.run(train_op, feed_dict={X: x, Y: y})
            print("x=", x, "y=", y)
    tf.summary.FileWriter(logDir, sess.graph).close()
    print(sess.run(w)) # It should be something around 2