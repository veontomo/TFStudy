import os
import tensorflow as tf
import numpy as np

logDir = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/") + "/log/"
print("Write the log to folder: " + logDir)
print("tensorflow version: " + tf.__version__)
L = 3 # dimension of the input vector
M = 15 # the number of the input vectors

# trX = [[0,2,0], [0,2,1]]
# trY = [1,0]
data = list(range(M))

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
	"1, if n is even, 0 otherwise"
	return 1 - (n % 2)

data2 = list(map(base3, data))
dataX = list(map(padding4, data2))
dataY = list(map(isEven, data))

X = tf.placeholder("int8", name="inputX") # create symbolic variables
Y = tf.placeholder("int8", name="inputY")

def model(X, w):
	return tf.matmul(tf.cast(X, tf.float64), w) # lr is just X*w so this model line is pretty simple

w = tf.Variable(np.random.rand(1, L), name='weight')
 # w = tf.Variable(0.0, name="weights") # create a shared variable (like theano.shared) for the weight matrix
y_model = model(X, w)

cost = tf.square(tf.cast(Y, tf.float64) - y_model) # use square error for cost function

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost) # construct an optimizer to minimize cost and fit line to my data

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize variables (in this case just variable W)
    tf.global_variables_initializer().run()

    for i in range(5):
        for (x, y) in zip(dataX, dataY):
            sess.run(train_op, feed_dict={X: x, Y: y})
            print(sess.run(w))
            print("x=", x, "y=", y)

    # tf.summary.FileWriter(logDir, sess.graph).close()
    # print(sess.run(w)) # It should be something around 2

# x0 = tf.Variable([[1, 0, 1]], name="init_data")
# x = tf.placeholder(tf.int8, shape=(1, N), name='input')
# x = x0
# w = tf.Variable(np.random.rand(M, 1), name='weight')
# z = tf.matmul(tf.cast(x, tf.float64), w)
# y = tf.nn.sigmoid(z, name="activation_fun")


# y_ = tf.constant([[0.0]])
# loss = tf.square(y - y_)
# optim = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# grads_and_vars = optim.compute_gradients(loss)
# train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)
# with tf.Session() as session:
 # tf.summary.FileWriter(logDir, session.graph).close()
 # session.run(tf.global_variables_initializer())
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
 # session.run(optim.apply_gradients(grads_and_vars))
 # for i in range(100):
  # session.run(train_step)
  # print(session.run(w))