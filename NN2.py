import os
import tensorflow as tf
import numpy as np

logDir = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/") + "/log/"
print("Write the log to folder: " + logDir)
print("tensorflow version: " + tf.__version__)
N = 3 # dimension of the input vector

x0 = tf.Variable([[1, 0, 1]], name="init_data")
# x = tf.placeholder(tf.int8, shape=(1, N), name='input')
x = x0
w = tf.Variable(np.random.rand(N, 1), name='weight')
z = tf.matmul(tf.cast(x, tf.float64), w)
y = tf.nn.sigmoid(z, name="activation_fun")


y_ = tf.constant([[0.0]])
loss = (y - y_)**2
optim = tf.train.GradientDescentOptimizer(learning_rate=0.01)
grads_and_vars = optim.compute_gradients(loss)
train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)
with tf.Session() as session:
 tf.summary.FileWriter(logDir, session.graph).close()
 session.run(tf.global_variables_initializer())
 # print(session.run(x))
 # print("x = ")
 # print(session.run(x))
 # print("w = ")
 # print(session.run(w))
 # print("z = ")
 # print(session.run(z))
 # print("y = ")
 # print(session.run(y))


 # print(session.run(grads_and_vars[0	][0]))
 session.run(optim.apply_gradients(grads_and_vars))
 for i in range(100):
  session.run(train_step)
  print(session.run(w))