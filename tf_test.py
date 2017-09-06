import tensorflow as tf
import numpy as np


x = tf.Variable(3.0)
y = tf.Variable(1.0)
 

f1 = tf.add(tf.square(tf.add(x, -y)), 1)
f2 = tf.add(tf.square(tf.add(x, y)), 1)

#tf.add_n([tf.square(tf.add(tf.subtract(1.5, x), tf.multiply(x, y))),
          #tf.square(tf.add(tf.subtract(2.25, x), tf.multiply(x, tf.square(y)))),
          #tf.square(tf.add(tf.subtract(2.625, x), tf.multiply(x, tf.pow(y, 3))))])


opt1 = tf.train.GradientDescentOptimizer(0.1).minimize(f1)
opt2 = tf.train.GradientDescentOptimizer(1.0).minimize(f2)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(3):
	    print("before", sess.run([x, y, f1, f2]))
	    sess.run(opt1)
	    print("after1", sess.run([x, y, f1, f2]))
	    sess.run(opt2)
	    print("after2", sess.run([x, y, f1, f2]))


print("opt1")
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(3):
	    print("before", sess.run([x, y, f1, f2]))
	    sess.run(opt1)
	    print("after1", sess.run([x, y, f1, f2]))



print("opt2")
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(3):
	    print("before", sess.run([x, y, f1, f2]))
	    sess.run(opt2)
	    print("after2", sess.run([x, y, f1, f2]))

X = tf.placeholder(tf.float32, (5, 3))
W = tf.placeholder(tf.float32, (1, 5))
Z = tf.matmul(W, X)
S = tf.sigmoid(Z)

with tf.Session() as s:
	print(s.run(S, feed_dict={X: np.random.randn(5, 3), W: np.randn.randn(1, 5)}))
