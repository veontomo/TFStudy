import tensorflow as tf
import numpy as np

# x = tf.Variable(3.0)
# y = tf.Variable(1.0)


# f1 = tf.add(tf.square(tf.add(x, -y)), 1)
# f2 = tf.add(tf.square(tf.add(x, y)), 1)

# tf.add_n([tf.square(tf.add(tf.subtract(1.5, x), tf.multiply(x, y))),
# tf.square(tf.add(tf.subtract(2.25, x), tf.multiply(x, tf.square(y)))),
# tf.square(tf.add(tf.subtract(2.625, x), tf.multiply(x, tf.pow(y, 3))))])


# opt1 = tf.train.GradientDescentOptimizer(0.1).minimize(f1)
# opt2 = tf.train.GradientDescentOptimizer(1.0).minimize(f2)

# with tf.Session() as sess:
#  sess.run(tf.global_variables_initializer())
# 	for i in range(3):
# 	    print("before", sess.run([x, y, f1, f2]))
# 	    sess.run(opt1)
# 	    print("after1", sess.run([x, y, f1, f2]))
# 	    sess.run(opt2)
# 	    print("after2", sess.run([x, y, f1, f2]))


# print("opt1")
# with tf.Session() as sess:
# 	sess.run(tf.global_variables_initializer())
# 	for i in range(3):
# 	    print("before", sess.run([x, y, f1, f2]))
# 	    sess.run(opt1)
# 	    print("after1", sess.run([x, y, f1, f2]))



# print("opt2")
# with tf.Session() as sess:
# 	sess.run(tf.global_variables_initializer())
# 	for i in range(3):
# 	    print("before", sess.run([x, y, f1, f2]))
# 	    sess.run(opt2)
# 	    print("after2", sess.run([x, y, f1, f2]))


X = tf.placeholder(tf.float32, (5, 3))
W = tf.Variable(tf.random_normal([1, 5], stddev=0.1))
Y = tf.constant(np.random.rand(1, 3))
Z = tf.matmul(W, X)
S = tf.sigmoid(Z)
loss = tf.losses.absolute_difference(Y, S)
opt = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as s:
    s.run(tf.global_variables_initializer())
    d = {X: np.random.rand(5, 3), Y: np.random.randn(1, 3)}
    for i in range(0, 10000):
        s.run(opt, feed_dict=d)
        if i % 200 == 0:
            print(i, s.run([W, loss], feed_dict=d))
