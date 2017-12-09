from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import time
from os.path import join as pjoin

#Numpy contains helpful functions for efficient mathematical calculations
import numpy as np
#Tensorflow library. Used to implement machine learning models
import tensorflow as tf
from data import fill_feed_dict_ae, read_data_sets_pretraining
from data import read_data_sets, fill_feed_dict
from flags import FLAGS
from flags import sub
from eval import loss_supervised, evaluation, do_eval_summary
from utils import tile_raster_images

#Import the math function for calculations
import math
#Image library for image manipulation
from PIL import Image
import sys
#Utils file
from utils import tile_raster_images

######
import pandas as pd

# python nids.py <RBM1> <RBM2> <RBM3> <RBMEPOCHS> <RBMLEARNRATE> <RNNLEARNRATE> <RNNEOPCHS> <TESTSET>

# PARAMETERS FOR RBM
param_rbm_hidden_sizes = []

if int(sys.argv[3]) == 0:
    # 2 layers
    param_rbm_hidden_sizes = [int(sys.argv[1]), int(sys.argv[2])]
else:
    # 3 layers
    param_rbm_hidden_sizes = [int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])]

param_rbm_epochs = int(sys.argv[4])
param_rbm_learning_rate = float(sys.argv[5])

# PARAMETERS FOR RNN/LSTM
param_rnnlstm_learning_rate = float(sys.argv[6])
param_rnnlstm_epochs = int(sys.argv[7])




# READING IN DATA AND CONVERTING TO 1HE: 
# http://pbpython.com/categorical-encoding.html
# http://fastml.com/converting-categorical-data-into-numbers-with-pandas-and-scikit-learn/

# # Define the headers since the data does not have any
headers = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","idk"]

# Read in the CSV file and convert "?" to NaN
df = pd.read_csv("KDDTrain+.csv",
                  header=None, names=headers, na_values="?" )

# print(df.head())
print(df.dtypes)

# Encode categorical variables e.g. protocol_type, service, flag, label
obj_df = df.select_dtypes(include=['object']).copy()
print(obj_df.head())

# Convert to columns of 0 or 1
# print(pd.get_dummies(obj_df, columns=["protocol_type", "service", "flag", "label"]).head(10))
df_with_dummies = pd.get_dummies(df, columns=["protocol_type", "service", "flag", "label"])

# https://stackoverflow.com/questions/19124601/is-there-a-way-to-pretty-print-the-entire-pandas-series-dataframe
with pd.option_context('display.max_rows', 3, 'display.max_columns', 146):
    # print(df_with_dummies) 
    print(df_with_dummies.shape) # (125973, 146)
    print(list(df_with_dummies))
# print(df_with_dummies)
# Convert pandas dataframe to numpy array https://stackoverflow.com/questions/13187778/convert-pandas-dataframe-to-numpy-array-preserving-index
# numpyMatrix = df_with_dummies.as_matrix()
# print(numpyMatrix)
# print(numpyMatrix.shape[1])

# df_X = df_with_dummies[['col_a', 'col_b']] data
# df_Y = df_with_dummies[['label_a','label_b']] classes
# get rid of that useless column  'num_outbound_cmds',
df_X = df_with_dummies[['duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'idk', 'protocol_type_icmp', 'protocol_type_tcp', 'protocol_type_udp', 'service_IRC', 'service_X11', 'service_Z39_50', 'service_aol', 'service_auth', 'service_bgp', 'service_courier', 'service_csnet_ns', 'service_ctf', 'service_daytime', 'service_discard', 'service_domain', 'service_domain_u', 'service_echo', 'service_eco_i', 'service_ecr_i', 'service_efs', 'service_exec', 'service_finger', 'service_ftp', 'service_ftp_data', 'service_gopher', 'service_harvest', 'service_hostnames', 'service_http', 'service_http_2784', 'service_http_443', 'service_http_8001', 'service_imap4', 'service_iso_tsap', 'service_klogin', 'service_kshell', 'service_ldap', 'service_link', 'service_login', 'service_mtp', 'service_name', 'service_netbios_dgm', 'service_netbios_ns', 'service_netbios_ssn', 'service_netstat', 'service_nnsp', 'service_nntp', 'service_ntp_u', 'service_other', 'service_pm_dump', 'service_pop_2', 'service_pop_3', 'service_printer', 'service_private', 'service_red_i', 'service_remote_job', 'service_rje', 'service_shell', 'service_smtp', 'service_sql_net', 'service_ssh', 'service_sunrpc', 'service_supdup', 'service_systat', 'service_telnet', 'service_tftp_u', 'service_tim_i', 'service_time', 'service_urh_i', 'service_urp_i', 'service_uucp', 'service_uucp_path', 'service_vmnet', 'service_whois', 'flag_OTH', 'flag_REJ', 'flag_RSTO', 'flag_RSTOS0', 'flag_RSTR', 'flag_S0', 'flag_S1', 'flag_S2', 'flag_S3', 'flag_SF', 'flag_SH']]
df_Y = df_with_dummies[['label_back', 'label_buffer_overflow', 'label_ftp_write', 'label_guess_passwd', 'label_imap', 'label_ipsweep', 'label_land', 'label_loadmodule', 'label_multihop', 'label_neptune', 'label_nmap', 'label_normal', 'label_perl', 'label_phf', 'label_pod', 'label_portsweep', 'label_rootkit', 'label_satan', 'label_smurf', 'label_spy', 'label_teardrop', 'label_warezclient', 'label_warezmaster']]
# convert
# list all columns
cols = list(df_X.columns)
print(cols)
for col in cols:
    # df_X[col] = (df_X[col] - df_X[col].mean())/df_X[col].std(ddof=0)
    df_X[col] = (df_X[col] - df_X[col].min())/(df_X[col].max() - df_X[col].min()) # map to 0 and 1
    # df_X[col] = df_X[col] * 255.0

print("After conversions")
# print(df_X)
# print(df_Y)
print(df_with_dummies.describe())

trX = df_X.as_matrix()
trY = df_Y.as_matrix()
trX = trX.astype(np.float32)
trY = trY.astype(np.float64)
print(trX.shape)
print(trY.shape)
print(trX.dtype)
print(trY.dtype)
# 122 features

# write into csv
# trX is a row, image converted into vector
# Take data and put into autoencoder
# in data processing, need to normalize data (zscore)
# some features are 0 and 1, others 0 - 1000
# prior to dummies, only applies to numeric data
# this mnist dataset is bounded from 0 - 1
# zscore doesn't actually do this

###RBM TIME BABY###

class RBM(object):

    def __init__(self, input_size, output_size):
        #Defining the hyperparameters
        self._input_size = input_size #Size of input
        self._output_size = output_size #Size of output
        self.epochs = param_rbm_epochs #Amount of training iterations orig: 25
        self.learning_rate = param_rbm_learning_rate #The step used in gradient descent orig: 1.0
        self.batchsize = 100 #The size of how much data will be used for training per sub iteration

        #Initializing weights and biases as matrices full of zeroes
        self.w = np.zeros([input_size, output_size], np.float32) #Creates and initializes the weights with 0
        self.hb = np.zeros([output_size], np.float32) #Creates and initializes the hidden biases with 0
        self.vb = np.zeros([input_size], np.float32) #Creates and initializes the visible biases with 0


    #Fits the result from the weighted visible layer plus the bias into a sigmoid curve
    def prob_h_given_v(self, visible, w, hb):
        #Sigmoid
        return tf.nn.sigmoid(tf.matmul(visible, w) + hb)

    #Fits the result from the weighted hidden layer plus the bias into a sigmoid curve
    def prob_v_given_h(self, hidden, w, vb):
        return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(w)) + vb)

    #Generate the sample probability
    def sample_prob(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

    #Training method for the model
    def train(self, X):
        #Create the placeholders for our parameters
        _w = tf.placeholder("float", [self._input_size, self._output_size])
        _hb = tf.placeholder("float", [self._output_size])
        _vb = tf.placeholder("float", [self._input_size])

        prv_w = np.zeros([self._input_size, self._output_size], np.float32) #Creates and initializes the weights with 0
        prv_hb = np.zeros([self._output_size], np.float32) #Creates and initializes the hidden biases with 0
        prv_vb = np.zeros([self._input_size], np.float32) #Creates and initializes the visible biases with 0


        cur_w = np.zeros([self._input_size, self._output_size], np.float32)
        cur_hb = np.zeros([self._output_size], np.float32)
        cur_vb = np.zeros([self._input_size], np.float32)
        v0 = tf.placeholder("float", [None, self._input_size])

        #Initialize with sample probabilities
        h0 = self.sample_prob(self.prob_h_given_v(v0, _w, _hb))
        v1 = self.sample_prob(self.prob_v_given_h(h0, _w, _vb))
        h1 = self.prob_h_given_v(v1, _w, _hb)

        #Create the Gradients
        positive_grad = tf.matmul(tf.transpose(v0), h0)
        negative_grad = tf.matmul(tf.transpose(v1), h1)

        #Update learning rates for the layers
        update_w = _w + self.learning_rate *(positive_grad - negative_grad) / tf.to_float(tf.shape(v0)[0])
        update_vb = _vb +  self.learning_rate * tf.reduce_mean(v0 - v1, 0)
        update_hb = _hb +  self.learning_rate * tf.reduce_mean(h0 - h1, 0)

        #Find the error rate
        err = tf.reduce_mean(tf.square(v0 - v1))

        #Training loop
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #For each epoch
            for epoch in range(self.epochs):
                #For each step/batch
                for start, end in zip(range(0, len(X), self.batchsize),range(self.batchsize,len(X), self.batchsize)):
                    batch = X[start:end]
                    #Update the rates
                    cur_w = sess.run(update_w, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    cur_hb = sess.run(update_hb, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    cur_vb = sess.run(update_vb, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    prv_w = cur_w
                    prv_hb = cur_hb
                    prv_vb = cur_vb
                error=sess.run(err, feed_dict={v0: X, _w: cur_w, _vb: cur_vb, _hb: cur_hb})
                print('Epoch: %d' % epoch,'reconstruction error: %f' % error)
            self.w = prv_w
            self.hb = prv_hb
            self.vb = prv_vb

    #Create expected output for our DBN
    def rbm_outpt(self, X):
        input_X = tf.constant(X)
        _w = tf.constant(self.w)
        _hb = tf.constant(self.hb)
        out = tf.nn.sigmoid(tf.matmul(input_X, _w) + _hb)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(out)

# print(trX.shape) # rows x features
# print(trY.shape) # classes. 10 digits
# print(trX.dtype) # float 32
# print(trY.dtype) # float 64

# RBM_hidden_sizes = [neurons1, neurons2]
# 50 epochs for each rbm
# 25 epochs for fine tuning
# RBM_hidden_sizes = [500, 200 , 50 ] #create 3 layers of RBM with size 500, 200, and 50
# RBM_hidden_sizes = [40,20,10]
# RBM_hidden_sizes = [10,5]
# RBM_hidden_sizes = [40,25]
# RBM_hidden_sizes = [100,25]
RBM_hidden_sizes = param_rbm_hidden_sizes # parameterized

# of last layer of RBM should be same as number of classes or greater

#Since we are training, set input as training data
inpX = trX

#Create list to hold our RBMs
rbm_list = []

#Size of inputs is the number of inputs in the training set
input_size = inpX.shape[1]
print("Input size:")
print(input_size)

#For each RBM we want to generate
for i, size in enumerate(RBM_hidden_sizes):
    print('RBM: ',i,' ',input_size,'->', size)
    rbm_list.append(RBM(input_size, size))
    input_size = size

#For each RBM in our list
for rbm in rbm_list:
    print('New RBM:')
    #Train a new one
    rbm.train(inpX) 
    #Return the output layer
    inpX = rbm.rbm_outpt(inpX)
    print(inpX.shape)

print(inpX.shape)
trX = inpX

class NN(object):

    def __init__(self, sizes, X, Y):
        #Initialize hyperparameters
        self._sizes = sizes
        self._X = X
        self._Y = Y
        self.w_list = []
        self.b_list = []
        self._learning_rate = param_rnnlstm_learning_rate # orig: 1.0
        self._training_iters = 100000
        self._display_step = 100 # orig: 10
        self._epoches = param_rnnlstm_epochs # orig: 25
        self._batchsize = 100
        input_size = X.shape[1]

        #initialization loop
        for size in self._sizes + [Y.shape[1]]:
            #Define upper limit for the uniform distribution range
            max_range = 4 * math.sqrt(6. / (input_size + size))

            #Initialize weights through a random uniform distribution
            self.w_list.append(
                np.random.uniform( -max_range, max_range, [input_size, size]).astype(np.float32))  # size = number of classes

            #Initialize bias as zeroes
            self.b_list.append(np.zeros([size], np.float32))
            input_size = size

    #load data from rbm
    def load_from_rbms(self, dbn_sizes,rbm_list):
        #Check if expected sizes are correct
        assert len(dbn_sizes) == len(self._sizes)

        for i in range(len(self._sizes)):
            #Check if for each RBN the expected sizes are correct
            assert dbn_sizes[i] == self._sizes[i]

        #If everything is correct, bring over the weights and biases
        for i in range(len(self._sizes)):
            self.w_list[i] = rbm_list[i].w
            self.b_list[i] = rbm_list[i].hb

    #Training method
    def train(self):
        # n_steps = self._X.shape[1]
        # n_steps = 5
        n_steps = 7
        n_input = self._X.shape[1] # 122    no should be 25. last layer
        n_classes = self._Y.shape[1] # 23   is number of classes
        n_hidden = n_input # orig: 25
        print("Classes") # should be 23 classes
        print(n_classes)
        #Create placeholders for input, weights, biases, output
        _a = [None] * (len(self._sizes) + 2)
        _w = [None] * (len(self._sizes) + 1)
        _b = [None] * (len(self._sizes) + 1)
        # _a[0] = tf.placeholder("float", [None, self._X.shape[1]])
        # _a[0] = tf.placeholder(dtype="float", shape=[None, n_steps, n_input], name="x") # (batchsize, steps, input) (?, 122, 122)
        # = to output rbm
        # inpX = []

        # get ouput from RBM
        # for rbm in rbm_list:
        #     #Return the output layer
        #     inpX = rbm.rbm_outpt(inpX)
        # inpX = self._X
        
        x = tf.placeholder(dtype="float", shape=[None, n_steps, 7], name="x") # 5 is supposed to be n_input
        y = tf.placeholder(dtype="float", shape=[None, n_classes], name="y") # num of classes
        print("inpX")
        print(self._X.shape)
        print("Num of features / hidden units. Size of last hidden layer")
        print(self._sizes[(len(self._sizes)-1)]) # hidden units num of features. how many hidden units to be in RBM?
        print("Y")
        print(y.shape) # rows x 23
        # n_hidden = self._sizes[(len(self._sizes)-1)]

        # Get weights and biases from RBM
        #Define variables and activation function
        for i in range(len(self._sizes) + 1):
            _w[i] = tf.Variable(self.w_list[i])
            _b[i] = tf.Variable(self.b_list[i])
        
        print("Weights:")
        print(_w)
        print("Biases:")
        print(_b)

        # Load in weights and biases from the last layer in the RBM
        weights = {
            'out': _w[(len(self._sizes))] # last layer
        }
        biases = {
            'out': _b[(len(self._sizes))] # last layer
        }
        print(weights['out'])
        print(biases['out'])

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0) # construct with hidden size of last layer in rbm

        # num_layers = 3
        # state_size = 100
        # embeddings = tf.get_variable('embedding_matrix', [n_classes, state_size])
        # rnn_inputs = [tf.squeeze(i) for i in tf.split(1,
        #                             n_steps, tf.nn.embedding_lookup(embeddings, x))]
        # cell = tf.nn.rnn_cell.LSTMCell(n_hidden, state_is_tuple=True)
        # cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

        # stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell] * num_layers)

        # initial_state = stacked_lstm.zero_state(self._batchsize, tf.float32)

        # outputs, states = tf.nn.dynamic_rnn(stacked_lstm, inputs=x, dtype=tf.float32)
        # outputs, states = tf.nn.dynamic_rnn(cell, inputs=rnn_inputs, dtype=tf.float32, initial_state=initial_state)
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, inputs=x, dtype=tf.float32)
        print(outputs) # (?, 122, 122)

        # reshaping, take 5 at a time
        # originally took 28 x 28
        # 1 x 28 at time
        # 5 pieces at a time
        # so reshape to 1 x 5

        # take vector, output into columns of 5

        # for i in range(1, len(self._sizes) + 2):
        output = tf.reshape(tf.split(outputs, n_steps, axis=1, num=None, name='split')[-1],[-1,n_hidden]) # x.shape is input
        print("Output")
        print(output)
        # print("Weight shape")
        # print(_w[i - 1].shape)
        pred = tf.matmul(output, weights['out']) + biases['out'] # linear activation to map it to a [?xclasses] [classes = 23]
        print(pred)
        # define cost function
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred ))
        # define training operation (adam optimizer minimizing cost function)
        optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(cost)
        print("optimizer")
        print(optimizer)

        # prediction operation
        correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            step = 0
            # test = 5000 # we use the last 5,000 as test
            test = 12000 # we test on 1/10th of the dataset
            sectionToTest = int(sys.argv[8])
            # Keep training until reach max iterations
            start = 0
            end = 100
            # while step * self._batchsize < self._training_iters:
            #For each epoch
            for i in range(self._epoches):
                # Test from 0 to lower bound of test section
                while (step * self._batchsize < len(self._X)-1825): # len(self._X):sectionToTest*test

                    # Skip test set
                    while (step * self._batchsize > test*sectionToTest) and (step * self._batchsize < test*(sectionToTest+1)):
                        start += 100
                        end += 100
                        step += 1

                    # print(len(self._X)-test)
                    batch_x = np.asarray(self._X[start:end]).reshape((self._batchsize, n_steps, 7))
                    batch_y = np.asarray(self._Y[start:end])
                    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                    if step % self._display_step == 0:
                        # Calculate batch accuracy
                        acc = sess.run(accuracy, feed_dict={
                        x: batch_x, y: batch_y})
                        # Calculate batch loss
                        loss = sess.run(cost, feed_dict={
                        x: batch_x, y: batch_y})
                        print("Iter " + str(step*self._batchsize) + ", Minibatch Loss= " + \
                            "{:.6f}".format(loss) + ", Training Accuracy= " + \
                            "{:.5f}".format(acc))

                    start += 100
                    end += 100
                    step += 1
                    

                    for j in range(len(self._sizes) + 1):
                        #Retrieve weights and biases
                        self.w_list[j] = sess.run(_w[j])
                        self.b_list[j] = sess.run(_b[j])

                print("Optimization Finished!")

                # Test
                start = sectionToTest*test
                end = (sectionToTest+1)*test
                test_data = np.asarray(self._X[start:end]).reshape((start-end, n_steps, 7)) # (batchsize, n_steps, 5)
                test_label = np.asarray(self._Y[start:end])
                
                #Run the training operation on the input data

                print("Testing Accuracy:", \
                sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

                # Reset to run next epoch
                step = 0
                start = 0
                end = 100

nNet = NN(RBM_hidden_sizes, trX, trY)
nNet.load_from_rbms(RBM_hidden_sizes,rbm_list)
nNet.train()


#Training method - Original shallow NN
    # def train(self):
    #     #Create placeholders for input, weights, biases, output
    #     _a = [None] * (len(self._sizes) + 2)
    #     _w = [None] * (len(self._sizes) + 1)
    #     _b = [None] * (len(self._sizes) + 1)
    #     _a[0] = tf.placeholder("float", [None, self._X.shape[1]])
    #     y = tf.placeholder("float", [None, self._Y.shape[1]])
        
    #     #Define variables and activation function
    #     for i in range(len(self._sizes) + 1):
    #         _w[i] = tf.Variable(self.w_list[i])
    #         _b[i] = tf.Variable(self.b_list[i])
    #     for i in range(1, len(self._sizes) + 2):
    #         _a[i] = tf.nn.sigmoid(tf.matmul(_a[i - 1], _w[i - 1]) + _b[i - 1])

    #     print(_w)
        
    #     #Define the cost function
    #     cost = tf.reduce_mean(tf.square(_a[-1] - y))
        
    #     #Define the training operation (Momentum Optimizer minimizing the Cost function)
    #     train_op = tf.train.MomentumOptimizer(
    #         self._learning_rate, self._momentum).minimize(cost)
        
    #     #Prediction operation
    #     predict_op = tf.argmax(_a[-1], 1)
        
    #     #Training Loop
    #     with tf.Session() as sess:
    #         #Initialize Variables
    #         sess.run(tf.global_variables_initializer())
            
    #         #For each epoch
    #         for i in range(self._epoches):
                
    #             #For each step
    #             for start, end in zip(
    #                 range(0, len(self._X), self._batchsize), range(self._batchsize, len(self._X), self._batchsize)):
                    
    #                 #Run the training operation on the input data
    #                 sess.run(train_op, feed_dict={
    #                     _a[0]: self._X[start:end], y: self._Y[start:end]})
                
    #             for j in range(len(self._sizes) + 1):
    #                 #Retrieve weights and biases
    #                 self.w_list[j] = sess.run(_w[j])
    #                 self.b_list[j] = sess.run(_b[j])
                
    #             print("Accuracy rating for epoch " + str(i) + ": " + str(np.mean(np.argmax(self._Y, axis=1) ==
    #                           sess.run(predict_op, feed_dict={_a[0]: self._X, y: self._Y}))))




                #For each step
                # for start, end in zip(
                #     range(0, len(self._X)-test, self._batchsize), range(self._batchsize, len(self._X)-test, self._batchsize)):
                    # range(0, len(self._X)-test, self._batchsize), range(self._batchsize, len(self._X)-test, self._batchsize)):
                    # print(start)
                    # print(end)
                    # print(np.asarray(self._X[start:end].shape))
                    # print(np.asarray(self._Y[start:end].shape))
                    # batch_x should be numpy array [batch_size, 25]
                    # reshape so we get 5 seq of 5 elements, batch_x should be [100x5x5]

                    # batch_x = np.asarray(self._X[start:end]).reshape((self._batchsize, n_steps, 5))

                    # print(batch_x)
                    # should be [batch_size, 23]

                    # batch_y = np.asarray(self._Y[start:end])

                    # print(batch_y)
                    
                    #Run the training operation on the input data
                    # sess.run(optimizer, feed_dict={
                    # x: batch_x, y: batch_y})

                    # if step % self._display_step == 0:
                    #     # # Calculate batch accuracy
                    #     # acc = sess.run(accuracy, feed_dict={
                    #     #     x[0]: self._X[start:end], y: self._Y[start:end]})
                    #     # # Calculate batch loss
                    #     # loss = sess.run(cost, feed_dict={
                    #     #     x[0]: self._X[start:end], y: self._Y[start:end]})
                    #     # Calculate batch accuracy
                    #     acc = sess.run(accuracy, feed_dict={
                    #     x: batch_x, y: batch_y})
                    #     # Calculate batch loss
                    #     loss = sess.run(cost, feed_dict={
                    #     x: batch_x, y: batch_y})
                    #     print("Iter " + str(step*self._batchsize) + ", Minibatch Loss= " + \
                    #         "{:.6f}".format(loss) + ", Training Accuracy= " + \
                    #         "{:.5f}".format(acc))

                    # step += 1

                # for j in range(len(self._sizes) + 1):
                #     #Retrieve weights and biases
                #     self.w_list[j] = sess.run(_w[j])
                #     self.b_list[j] = sess.run(_b[j])
                
                

            # print("Optimization Finished!")

            # for start, end in zip(
            #     range(len(self._X)-test, len(self._X), self._batchsize), range(self._batchsize, len(self._X), self._batchsize)):
            #     test_data = np.asarray(self._X[start:end]).reshape((self._batchsize, n_steps, 5))
            #     test_label = np.asarray(self._Y[start:end])
            # while step * self._batchsize < len(self._X):
            # start = len(self._X)-test
            # end = len(self._X)
            # print(start)
            # print(end)
            # # test_data = np.asarray(self._X[start:end]).reshape((self._batchsize, n_steps, 5))
            # test_data = np.asarray(self._X[start:end]).reshape((start-end, n_steps, 5))
            # test_label = np.asarray(self._Y[start:end])
            
            # #Run the training operation on the input data

            # print("Testing Accuracy:", \
            # sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
                # start += 100
                # end += 100
                # step += 1

            # Calculate accuracy on test data
            # load in test data... have to perform transformations again

            # headers = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","idk"]

            # # Read in the CSV file and convert "?" to NaN
            # test_df = pd.read_csv("KDDTest+.csv",
            #                 header=None, names=headers, na_values="?" )

            # # Encode categorical variables e.g. protocol_type, service, flag, label
            # test_obj_df = df.select_dtypes(include=['object']).copy()

            # # Convert to columns of 0 or 1
            # test_df_with_dummies = pd.get_dummies(df, columns=["protocol_type", "service", "flag", "label"])

            # test_df_X = test_df_with_dummies[['duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'idk', 'protocol_type_icmp', 'protocol_type_tcp', 'protocol_type_udp', 'service_IRC', 'service_X11', 'service_Z39_50', 'service_aol', 'service_auth', 'service_bgp', 'service_courier', 'service_csnet_ns', 'service_ctf', 'service_daytime', 'service_discard', 'service_domain', 'service_domain_u', 'service_echo', 'service_eco_i', 'service_ecr_i', 'service_efs', 'service_exec', 'service_finger', 'service_ftp', 'service_ftp_data', 'service_gopher', 'service_harvest', 'service_hostnames', 'service_http', 'service_http_2784', 'service_http_443', 'service_http_8001', 'service_imap4', 'service_iso_tsap', 'service_klogin', 'service_kshell', 'service_ldap', 'service_link', 'service_login', 'service_mtp', 'service_name', 'service_netbios_dgm', 'service_netbios_ns', 'service_netbios_ssn', 'service_netstat', 'service_nnsp', 'service_nntp', 'service_ntp_u', 'service_other', 'service_pm_dump', 'service_pop_2', 'service_pop_3', 'service_printer', 'service_private', 'service_red_i', 'service_remote_job', 'service_rje', 'service_shell', 'service_smtp', 'service_sql_net', 'service_ssh', 'service_sunrpc', 'service_supdup', 'service_systat', 'service_telnet', 'service_tftp_u', 'service_tim_i', 'service_time', 'service_urh_i', 'service_urp_i', 'service_uucp', 'service_uucp_path', 'service_vmnet', 'service_whois', 'flag_OTH', 'flag_REJ', 'flag_RSTO', 'flag_RSTOS0', 'flag_RSTR', 'flag_S0', 'flag_S1', 'flag_S2', 'flag_S3', 'flag_SF', 'flag_SH']]
            # test_df_Y = test_df_with_dummies[['label_back', 'label_buffer_overflow', 'label_ftp_write', 'label_guess_passwd', 'label_imap', 'label_ipsweep', 'label_land', 'label_loadmodule', 'label_multihop', 'label_neptune', 'label_nmap', 'label_normal', 'label_perl', 'label_phf', 'label_pod', 'label_portsweep', 'label_rootkit', 'label_satan', 'label_smurf', 'label_spy', 'label_teardrop', 'label_warezclient', 'label_warezmaster']]
            # test_cols = list(test_df_X.columns)
            # for col in test_cols:
            #     test_df_X[col] = (test_df_X[col] - test_df_X[col].min())/(test_df_X[col].max() - test_df_X[col].min()) # map to 0 and 1

            # test_trX = test_df_X.as_matrix()
            # test_trY = test_df_Y.as_matrix()
            # test_trX = test_trX.astype(np.float32)
            # test_trY = test_trY.astype(np.float64)

            # How2getaccuracy?
            # Test is labeled

            # for start, end in zip(
            #     range(len(self._X)-test, len(self._X), self._batchsize), range(self._batchsize, len(self._X), self._batchsize)):
            #     test_data = np.asarray(self._X[start:end]).reshape((self._batchsize, n_steps, 5))
            #     test_label = np.asarray(self._Y[start:end])
                
            #     #Run the training operation on the input data

            #     print("Testing Accuracy:", \
            #     sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

            # # Calculate accuracy for 128 mnist test images
            # test_len = 128
            # test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
            # test_label = mnist.test.labels[:test_len]
            # print("Testing Accuracy:", \
            #     sess.run(accuracy, feed_dict={x: test_data, y: test_label}))


# nNet = NN(RBM_hidden_sizes, trX, trY)
# nNet.load_from_rbms(RBM_hidden_sizes,rbm_list)
# nNet.train()



############################################autoencoder.py############################################################
# Tensorflow deep autoencoder implementation with summaries
# originally by @cmgreen210
# extensively modified by @pablorp80 
# tested on tensorflow 1.3.0 with GPU and without 
# over ubuntu 16 and macos
# cuda 8

# class AutoEncoder(object):
#   """Generic deep autoencoder.

#   Autoencoder used for full training cycle, including
#   unsupervised pretraining layers and final fine tuning.
#   The user specifies the structure of the neural net
#   by specifying number of inputs, the number of hidden
#   units for each layer and the number of final output
#   logits.
#   """
#   _weights_str = "weights{0}"
#   _biases_str = "biases{0}"

#   def __init__(self, shape, sess):
#     """Autoencoder initializer

#     Args:
#       shape: list of ints specifying
#               num input, hidden1 units,...hidden_n units, num logits
#       sess: tensorflow session object to use
#     """
#     self.__shape = shape  # [input_dim,hidden1_dim,...,hidden_n_dim,output_dim]
#     self.__num_hidden_layers = len(self.__shape) - 2

#     self.__variables = {}
#     self.__sess = sess

#     self._setup_variables()

#   @property
#   def shape(self):
#     return self.__shape

#   @property
#   def num_hidden_layers(self):
#     return self.__num_hidden_layers

#   @property
#   def session(self):
#     return self.__sess

#   def __getitem__(self, item):
#     """Get autoencoder tf variable

#     Returns the specified variable created by this object.
#     Names are weights#, biases#, biases#_out, weights#_fixed,
#     biases#_fixed.

#     Args:
#      item: string, variables internal name
#     Returns:
#      Tensorflow variable
#     """
#     return self.__variables[item]

#   def __setitem__(self, key, value):
#     """Store a tensorflow variable

#     NOTE: Don't call this explicity. It should
#     be used only internally when setting up
#     variables.

#     Args:
#       key: string, name of variable
#       value: tensorflow variable
#     """
#     self.__variables[key] = value

#   def _setup_variables(self):
#     with tf.name_scope("autoencoder_variables"):
#       for i in range(self.__num_hidden_layers + 1):
#         # Train weights
#         name_w = self._weights_str.format(i + 1)
#         w_shape = (self.__shape[i], self.__shape[i + 1])
#         a = tf.multiply(4.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))
#         w_init = tf.random_uniform(w_shape, -1 * a, a)
#         self[name_w] = tf.Variable(w_init,
#                                    name=name_w,
#                                    trainable=True)
#         # Train biases
#         name_b = self._biases_str.format(i + 1)
#         b_shape = (self.__shape[i + 1],)
#         b_init = tf.zeros(b_shape)
#         self[name_b] = tf.Variable(b_init, trainable=True, name=name_b)

#         if i < self.__num_hidden_layers:
#           # Hidden layer fixed weights (after pretraining before fine tuning)
#           self[name_w + "_fixed"] = tf.Variable(tf.identity(self[name_w]),
#                                                 name=name_w + "_fixed",
#                                                 trainable=False)

#           # Hidden layer fixed biases
#           self[name_b + "_fixed"] = tf.Variable(tf.identity(self[name_b]),
#                                                 name=name_b + "_fixed",
#                                                 trainable=False)

#           # Pretraining output training biases
#           name_b_out = self._biases_str.format(i + 1) + "_out"
#           b_shape = (self.__shape[i],)
#           b_init = tf.zeros(b_shape)
#           self[name_b_out] = tf.Variable(b_init,
#                                          trainable=True,
#                                          name=name_b_out)

#   def _w(self, n, suffix=""):
#     return self[self._weights_str.format(n) + suffix]

#   def _b(self, n, suffix=""):
#     return self[self._biases_str.format(n) + suffix]

#   def get_variables_to_init(self, n):
#     """Return variables that need initialization

#     This method aides in the initialization of variables
#     before training begins at step n. The returned
#     list should be than used as the input to
#     tf.initialize_variables

#     Args:
#       n: int giving step of training
#     """
#     assert n > 0
#     assert n <= self.__num_hidden_layers + 1

#     vars_to_init = [self._w(n), self._b(n)]

#     if n <= self.__num_hidden_layers:
#       vars_to_init.append(self._b(n, "_out"))

#     if 1 < n <= self.__num_hidden_layers:
#       vars_to_init.append(self._w(n - 1, "_fixed"))
#       vars_to_init.append(self._b(n - 1, "_fixed"))

#     return vars_to_init

#   @staticmethod
#   def _activate(x, w, b, transpose_w=False):
#     y = tf.sigmoid(tf.nn.bias_add(tf.matmul(x, w, transpose_b=transpose_w), b))
#     return y

#   def pretrain_net(self, input_pl, n, is_target=False):
#     """Return net for step n training or target net

#     Args:
#       input_pl:  tensorflow placeholder of AE inputs
#       n:         int specifying pretrain step
#       is_target: bool specifying if required tensor
#                   should be the target tensor
#     Returns:
#       Tensor giving pretraining net or pretraining target
#     """
#     assert n > 0
#     assert n <= self.__num_hidden_layers

#     last_output = input_pl
#     for i in range(n - 1):
#       w = self._w(i + 1, "_fixed")
#       b = self._b(i + 1, "_fixed")

#       last_output = self._activate(last_output, w, b)

#     if is_target:
#       return last_output

#     last_output = self._activate(last_output, self._w(n), self._b(n))

#     out = self._activate(last_output, self._w(n), self._b(n, "_out"),
#                          transpose_w=True)
#     out = tf.maximum(out, 1.e-9)
#     out = tf.minimum(out, 1 - 1.e-9)
#     return out

#   def supervised_net(self, input_pl):
#     """Get the supervised fine tuning net

#     Args:
#       input_pl: tf placeholder for ae input data
#     Returns:
#       Tensor giving full ae net
#     """
#     last_output = input_pl

#     for i in range(self.__num_hidden_layers + 1):
#       # Fine tuning will be done on these variables
#       w = self._w(i + 1)
#       b = self._b(i + 1)

#       last_output = self._activate(last_output, w, b)

#     return last_output


# loss_summaries = {}


# def training(loss, learning_rate, loss_key=None):
#   """Sets up the training Ops.

#   Creates a summarizer to track the loss over time in TensorBoard.

#   Creates an optimizer and applies the gradients to all trainable variables.

#   The Op returned by this function is what must be passed to the
#   `sess.run()` call to cause the model to train.

#   Args:
#     loss: Loss tensor, from loss().
#     learning_rate: The learning rate to use for gradient descent.
#     loss_key: int giving stage of pretraining so we can store
#                 loss summaries for each pretraining stage

#   Returns:
#     train_op: The Op for training.
#   """
#   if loss_key is not None:
#     # Add a scalar summary for the snapshot loss.
#     loss_summaries[loss_key] = tf.summary.scalar(loss.op.name, loss)
#   else:
#     tf.summary.scalar(loss.op.name, loss)
#     for var in tf.trainable_variables():
#       tf.summary.histogram(var.op.name, var)
#   # Create the gradient descent optimizer with the given learning rate.
#   optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#   # Create a variable to track the global step.
#   global_step = tf.Variable(0, name='global_step', trainable=False)
#   # Use the optimizer to apply the gradients that minimize the loss
#   # (and also increment the global step counter) as a single training step.
#   train_op = optimizer.minimize(loss, global_step=global_step)
#   return train_op, global_step


# def loss_x_entropy(output, target):
#   """Cross entropy loss

#   See https://en.wikipedia.org/wiki/Cross_entropy

#   Args:
#     output: tensor of net output
#     target: tensor of net we are trying to reconstruct
#   Returns:
#     Scalar tensor of cross entropy
#   """
#   with tf.name_scope("xentropy_loss"):
#       net_output_tf = tf.convert_to_tensor(output, name='input')
#       target_tf = tf.convert_to_tensor(target, name='target')
#       cross_entropy = tf.add(tf.multiply(tf.log(net_output_tf, name='log_output'),
#                                     target_tf),
#                              tf.multiply(tf.log(1 - net_output_tf),
#                                     (1 - target_tf)))
#       return -1 * tf.reduce_mean(tf.reduce_sum(cross_entropy, 1),
#                                  name='xentropy_mean')


# def main_unsupervised():
#   with tf.Graph().as_default() as g:
#     sess = tf.Session()

#     # num_hidden = FLAGS.num_hidden_layers
#     num_hidden = 1
#     # ae_hidden_shapes = [getattr(FLAGS, "hidden{0}_units".format(j + 1))
#     #                     for j in range(num_hidden)]
#     ae_hidden_shapes = [100]
#     # ae_shape = [FLAGS.image_pixels] + ae_hidden_shapes + [FLAGS.num_classes]
#     ae_shape = [numpyMatrix.shape[1]] + ae_hidden_shapes + [FLAGS.num_classes] # of attack types: 24 
#     print("HEY")
#     print(ae_shape)

#     ae = AutoEncoder(ae_shape, sess)

#     # data = read_data_sets_pretraining(FLAGS.data_dir, 
#     #                                   sub['tr'], sub['te'], sub['val'], 
#     #                                   one_hot=True)
#     data = numpyMatrix

#     # num_train = data.train.num_examples
#     num_train = numpyMatrix.shape[1]

#     # learning_rates = {j: float(getattr(FLAGS,
#     #                              "pre_layer{0}_learning_rate".format(j + 1)))
#     #                   for j in range(num_hidden)}
#     learning_rates = {0 : 0.001, 1 : 0.001}

#     noise = {j: getattr(FLAGS, "noise_{0}".format(j + 1))
#              for j in range(num_hidden)}

#     for i in range(len(ae_shape) - 2):
#       n = i + 1
#       with tf.variable_scope("pretrain_{0}".format(n)):
#         input_ = tf.placeholder(dtype=tf.float32,
#                                 shape=(FLAGS.batch_size, ae_shape[0]),
#                                 name='ae_input_pl')
#         target_ = tf.placeholder(dtype=tf.float32,
#                                  shape=(FLAGS.batch_size, ae_shape[0]),
#                                  name='ae_target_pl')
#         layer = ae.pretrain_net(input_, n)

#         with tf.name_scope("target"):
#           target_for_loss = ae.pretrain_net(target_, n, is_target=True)

#         loss = loss_x_entropy(layer, target_for_loss)
#         train_op, global_step = training(loss, learning_rates[i], i)

#         summary_dir = pjoin(FLAGS.summary_dir, 'pretraining_{0}'.format(n))
#         summary_writer = tf.summary.FileWriter(summary_dir,
#                                                 graph=sess.graph,
#                                                 flush_secs=FLAGS.flush_secs)
#         summary_vars = [ae["biases{0}".format(n)], ae["weights{0}".format(n)]]

#         hist_summarries = [tf.summary.histogram(v.op.name, v)
#                            for v in summary_vars]
#         hist_summarries.append(loss_summaries[i])
#         summary_op = tf.summary.merge(hist_summarries)

#         vars_to_init = ae.get_variables_to_init(n)
#         vars_to_init.append(global_step)
#         sess.run(tf.variables_initializer(vars_to_init))

#         print("\n\n")
#         print("| Training Step | Cross Entropy |  Layer  |   Epoch  |")
#         print("|---------------|---------------|---------|----------|")

#         for step in range(FLAGS.pretraining_epochs * num_train):
#         #   feed_dict = fill_feed_dict_ae(data.train, input_, target_, noise[i])
#           feed_dict = fill_feed_dict_ae(data, input_, target_, noise[i])

#           loss_summary, loss_value = sess.run([train_op, loss],
#                                               feed_dict=feed_dict)

#           if step % 30 == 0:
#             summary_str = sess.run(summary_op, feed_dict=feed_dict)
#             summary_writer.add_summary(summary_str, step)
#             image_summary_op = \
#                 tf.summary.image("training_images",
#                                  tf.reshape(input_,
#                                             (FLAGS.batch_size,
#                                              FLAGS.image_size,
#                                              FLAGS.image_size, 1)),
#                                  max_outputs=FLAGS.batch_size)

#             summary_img_str = sess.run(image_summary_op,
#                                        feed_dict=feed_dict)
#             summary_writer.add_summary(summary_img_str)

#             output = "| {0:>13} | {1:13.4f} | Layer {2} | Epoch {3}  |"\
#                      .format(step, loss_value, n, step // num_train + 1)

#             print(output)
#       if i == 0:
#         filters = sess.run(tf.identity(ae["weights1"]))
#         np.save(pjoin(FLAGS.chkpt_dir, "filters"), filters)
#         filters = tile_raster_images(X=filters.T,
#                                      img_shape=(FLAGS.image_size,
#                                                 FLAGS.image_size),
#                                      tile_shape=(10, 10),
#                                      output_pixel_vals=False)
#         filters = np.expand_dims(np.expand_dims(filters, 0), 3)
#         image_var = tf.Variable(filters)
#         image_filter = tf.identity(image_var)
#         sess.run(tf.variables_initializer([image_var]))
#         img_filter_summary_op = tf.summary.image("first_layer_filters",
#                                                  image_filter)
#         summary_writer.add_summary(sess.run(img_filter_summary_op))
#         summary_writer.flush()

#   return ae


# def main_supervised(ae):
#   with ae.session.graph.as_default():
#     sess = ae.session
#     # input_pl = tf.placeholder(tf.float32, shape=(FLAGS.batch_size,
#     #                                              FLAGS.image_pixels),
#     #                           name='input_pl')
#     # logits = ae.supervised_net(input_pl)

#     # data = read_data_sets(FLAGS.data_dir, 
#     #                       sub['tr'], sub['te'], sub['val'],
#     #                       one_hot=False)
#     # data is in the form of dataframe

#     # num_train = data.train.num_examples
#     input_pl = tf.placeholder(tf.float32, shape=(FLAGS.batch_size,
#                                                  numpyMatrix.shape[1]),
#                               name='input_pl')
#     logits = ae.supervised_net(input_pl)

#     num_hidden = 1
#     ae_hidden_shapes = [100]
#     ae_shape = [numpyMatrix.shape[1]] + ae_hidden_shapes

#     # ae = AutoEncoder(ae_shape, sess)

#     data = numpyMatrix

#     num_train = numpyMatrix[1]

#     labels_placeholder = tf.placeholder(tf.int32,
#                                         shape=FLAGS.batch_size,
#                                         name='target_pl')

#     loss = loss_supervised(logits, labels_placeholder)
#     train_op, global_step = training(loss, FLAGS.supervised_learning_rate)
#     eval_correct = evaluation(logits, labels_placeholder)

#     hist_summaries = [ae['biases{0}'.format(i + 1)]
#                       for i in range(ae.num_hidden_layers + 1)]
#     hist_summaries.extend([ae['weights{0}'.format(i + 1)]
#                            for i in range(ae.num_hidden_layers + 1)])

#     hist_summaries = [tf.summary.histogram(v.op.name + "_fine_tuning", v)
#                       for v in hist_summaries]
#     summary_op = tf.summary.merge(hist_summaries)

#     summary_writer = tf.summary.FileWriter(pjoin(FLAGS.summary_dir,
#                                                   'fine_tuning'),
#                                             graph=sess.graph,
#                                             flush_secs=FLAGS.flush_secs)

#     vars_to_init = ae.get_variables_to_init(ae.num_hidden_layers + 1)
#     vars_to_init.append(global_step)
#     sess.run(tf.variables_initializer(vars_to_init))

#     steps = FLAGS.finetuning_epochs * num_train
    
#     for step in range(steps):
#       start_time = time.time()

#     #   feed_dict = fill_feed_dict(data.train,
#       feed_dict = fill_feed_dict(data.train,
#                                  input_pl,
#                                  labels_placeholder)

#       _, loss_value = sess.run([train_op, loss],
#                                feed_dict=feed_dict)

#       duration = time.time() - start_time

#       # Write the summaries and print an overview fairly often.
#       if step % 30 == 0:
#         # Print status to stdout.
#         print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
#         # Update the events file.

#         summary_str = sess.run(summary_op, feed_dict=feed_dict)
#         summary_writer.add_summary(summary_str, step)
#         summary_img_str = sess.run(
#             tf.summary.image("training_images",
#                              tf.reshape(input_pl,
#                                         (FLAGS.batch_size,
#                                          FLAGS.image_size,
#                                          FLAGS.image_size, 1)),
#                              max_outputs=FLAGS.batch_size),
#             feed_dict=feed_dict
#         )
#         summary_writer.add_summary(summary_img_str)

#       if (step + 1) % 30 == 0 or (step + 1) == steps:
#         train_sum = do_eval_summary("training_error",
#                                     sess,
#                                     eval_correct,
#                                     input_pl,
#                                     labels_placeholder,
#                                     data.train)

#         val_sum = do_eval_summary("validation_error",
#                                   sess,
#                                   eval_correct,
#                                   input_pl,
#                                   labels_placeholder,
#                                   data.validation)

#         test_sum = do_eval_summary("test_error",
#                                    sess,
#                                    eval_correct,
#                                    input_pl,
#                                    labels_placeholder,
#                                    data.test)

#         summary_writer.add_summary(train_sum, step)
#         summary_writer.add_summary(val_sum, step)
#         summary_writer.add_summary(test_sum, step)

# if __name__ == '__main__':
#   ae = main_unsupervised()
# #   main_supervised(ae)

