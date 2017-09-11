# sup biatch
# By Wengy i like buttttsssss 3
import numpy as np
import tensorflow as tf
print("hi theere sexy man")
a = np.array([[1,0],[0,1]])
print(a)
print(a.shape)
b = np.array([[5,6],[7,8.0]])
print(b)
print(b.shape)

a = tf.constant(a.astype(float))
b = tf.constant(b)
with tf.Session() as sess:
	print(tf.matmul(a,b).eval())