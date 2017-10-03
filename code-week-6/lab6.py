import numpy as np
import tensorflow as tf
sess = tf.Session()

#every layer in RNN will have own cell

#initialize a vector state
LSTM_CELL_SIZE = 4  # output size (dimension), which is same as hidden size in the cell

lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_CELL_SIZE, state_is_tuple=True)
state = (tf.zeros([2,LSTM_CELL_SIZE]),)*2 #initialized with zeroes
print(state) #tuple of (2, 4). 1x2. previous and next.

#lstm cell contains basic cell we explained earlier. read, write, memory
#state that is tuple that contains previous state and following state
#each with 4 elements

#now let's put some sample input
sample_input = tf.constant([[1,2,3,4,3,2],[3,2,2,2,2,2]],dtype=tf.float32)
print(sess.run(sample_input)) #vector representation of text/image. sequence/vector representation of things.

#pass input to lstm cell
with tf.variable_scope("LSTM_sample1"):
    output, state_new = lstm_cell(sample_input, state) #lstm cell will receive as input sample_input (which is vector) and state (has everything in previous state, and someplace for current state)
    # returns output, and that new state
sess.run(tf.global_variables_initializer()) #initialize global variables
print(sess.run(state_new)) #printing output and information about new state
# c is previous state, h is previous output
# 4 params, corresponds to the gates
print (sess.run(output))
# diff between state and output: state reps information about what things are at the moment, snapshot of what the LSTM looks like at the moment
# previous output represents the arrows connecting between neurons (they're passing state of parameters)
# LSTM fits into part of network where it connects to the next h
# to sum up: prv_output is h. LSTM contains state.

