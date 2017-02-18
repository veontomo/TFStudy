import os
import tensorflow as tf

print("tensorflow version: " + tf.__version__)
logDir = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/") + "/log/"
print("Write the log to folder: " + logDir)

x = tf.constant(1.0, name='input')
w = tf.Variable(0.8, name='weight')
y = tf.multiply(w, x, name='output')

y_ = tf.constant(0.0)
loss = (y - y_)**2
optim = tf.train.GradientDescentOptimizer(learning_rate=0.01)
grads_and_vars = optim.compute_gradients(loss)
train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)
with tf.Session() as session:
 writer = tf.summary.FileWriter(logDir, session.graph)
 writer.close()
 session.run(tf.global_variables_initializer())
 print(session.run(grads_and_vars[0	][0]))
 session.run(optim.apply_gradients(grads_and_vars))
 for i in range(100):
  session.run(train_step)
  print(session.run(w))