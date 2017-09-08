import numpy as np
import tensorflow as tf

a = np.array([[1,2],[4,-1],[-3,3]])
print("Array A transposed: ")
print(a)
b = np.array([[-2, 0, 5], [0, -1, 4]])
print("Array B: ")
print(b)

a = tf.constant(a)
b = tf.constant(b)

with tf.Session() as sess:
    t = tf.matmul(a,b).eval()
    print("Multiplication of matrices: ")
    print(t)
    print("Rank: (using numpy)")
    print(np.linalg.matrix_rank(t))
    print("Rank: (using tensorflow)")
    print(tf.rank(t).eval())

    ## Note on TensorFlow.org's api doc for tf.rank...
    # The rank of a tensor is not the same as the rank of a matrix.
    # The rank of a tensor is the number of indices required to uniquely select each element of the tensor.
    # Rank is also known as "order", "degree", or "ndims."
    # https://www.tensorflow.org/api_docs/python/tf/rank