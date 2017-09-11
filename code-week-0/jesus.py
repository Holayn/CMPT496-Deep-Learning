# jesus.py

import numpy as np
import tensorflow as tf

# model parameters
m = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

#model input & output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

linear_model = m * x + b

#define loss function (sum of squares)
loss = tf.reduce_sum(tf.square(linear_model - y))

#define optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

#training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong

curr_loss = float('inf')
while(curr_loss > 0.1):
	sess.run(train, {x:x_train, y:y_train})
	curr_m, curr_b, curr_loss = sess.run([m, b, loss], {x:x_train, y:y_train})
	print("m: %s, b: %s, loss: %s"%(curr_m, curr_b, curr_loss))