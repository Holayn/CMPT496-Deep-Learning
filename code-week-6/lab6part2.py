import numpy as np
import tensorflow as tf
sess = tf.Session()

#stacked lstm

LSTM_CELL_SIZE = 4  #4 hidden nodes = state_dim = the output_dim 
input_dim = 6
num_layers = 2 # 2 now

cells = []
for _ in range(num_layers):
    cell = tf.contrib.rnn.LSTMCell(LSTM_CELL_SIZE)
    cells.append(cell)
stacked_lstm = tf.contrib.rnn.MultiRNNCell(cells) #create multiple RNN cell

# Batch size x time steps x features.
# dynamic RNN. placeholder = variable used as input.
data = tf.placeholder(tf.float32, [None, None, input_dim])
output, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32) #using cell, data, floating type
# use it to feed to next place

#Batch size x time steps x features.
# Lets say the input sequence length is 3, and the dimensionality of the inputs is 6. The input should be a Tensor of shape: [batch_size, max_time, dimension], in our case it would be (2, 3, 6)
sample_input = [[[1,2,3,4,3,2], [1,2,1,1,1,2],[1,2,2,2,2,2]],[[1,2,3,4,3,2],[3,2,2,1,1,2],[0,0,0,0,3,2]]]
print(sample_input)
# iteration 1, we pass first item in sequence. observe, retrieve and feed to next, which will provide with next prediction.

sess.run(tf.global_variables_initializer())
sess.run(output, feed_dict={data: sample_input}) # run runs part of graph that evaluating
# feed data as simple input (line 20)
