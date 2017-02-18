import os
import tensorflow as tf

print("tensorflow version: " + tf.__version__)
logDir = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/") + "/log/"
print("Write the log to folder: " + logDir)

a = tf.add(1, 2,)
b = tf.multiply(a, 3)
c = tf.add(4, 5,)
d = tf.multiply(c, 6,)
e = tf.multiply(4, 5,)
f = tf.div(c, 6,)
g = tf.add(b, d)
h = tf.multiply(g, f)
with tf.Session() as session:
	writer = tf.summary.FileWriter(logDir, session.graph)
	print(session.run(h))
	writer.close()

# once the code above is executed, the log folder should contain a file with data about the calculation graph.
# In order to visualize it, run 
# tensorboard --logdir=path/to/log-directory