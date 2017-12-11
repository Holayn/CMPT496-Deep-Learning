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
        
        x = tf.placeholder(dtype="float", shape=[None, n_steps, 7], name="x") # 5 is supposed to be n_input
        y = tf.placeholder(dtype="float", shape=[None, n_classes], name="y") # num of classes
        print("inpX")
        print(self._X.shape)
        print("Num of features / hidden units. Size of last hidden layer")
        print(self._sizes[(len(self._sizes)-1)]) # hidden units num of features. how many hidden units to be in RBM?
        print("Y")
        print(y.shape) # rows x 23

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
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, inputs=x, dtype=tf.float32)
        print(outputs) # (?, 122, 122)

        # reshaping, take 5 at a time
        # originally took 28 x 28
        # 1 x 28 at time
        # 5 pieces at a time
        # so reshape to 1 x 5

        # take vector, output into columns of 5

        output = tf.reshape(tf.split(outputs, n_steps, axis=1, num=None, name='split')[-1],[-1,n_hidden]) # x.shape is input
        print("Output")
        print(output)
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
