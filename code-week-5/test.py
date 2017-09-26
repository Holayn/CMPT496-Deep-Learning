'''How to load own data into tensorflow'''

import tensorflow as tf

#Important object/class, called Dataset, it's a pain in the ass
#But if can make it work/set up correctly, saves a lot of pain with setting
#up datasets in the future. It's a big deal
from tensorflow.contrib.data import Dataset, Iterator

# we have two classes

NUM_CLASSES = 2

#important for reading process of image
def input_parser(img_path, label):
    # convert the label to one-hot encoding - we have two classes, so we have two "cables" [1 0] is spiral, [0 1] is twin.
    # vector representing what class it is important for outputting probabilities

    #label is data that want to transform
    #num class is depth (how many classes you have, size of vector for every single label)
    one_hot = tf.one_hot(label, NUM_CLASSES)

    #read images from disk
    #can also resize image in tf so that everything is the same
    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_image(img_file, channels=3)

    #call tensorflow's method to resize image, will interpolate to desired size
    #have to do it because all images have different sizes

    return img_decoded, one_hot

# galaxy data
train_imgs = tf.constant(['train/img1.jpg', 'train/img2.jpg', 
                          'train/img4.jpg', 'train/img5.jpg'])
#We will have two classes: classes that refer to spirals are class 0, and classes that refer to twin galaxies are class 1
#They are like the desired outputs
train_labels = tf.constant([0, 0, 1, 1])

test_imgs = tf.constant(['test/img3.jpg', 'test/img6.jpg'])

test_labels = tf.constant([0, 1])

# tf dataset object
#Creating the dataset object
#Read in train images as tuple
#This creates a tensor that is ready to be sliced
#Important in tensorflow when it begins training iterations
#Label and path to image
tr_data = Dataset.from_tensor_slices((train_imgs, train_labels))
te_data = Dataset.from_tensor_slices((test_imgs, test_labels))

#After this transformation, transforming data into tangible 0's to 255's
#calling itself on itself, so input_parser gets its arguments
tr_data = tr_data.map(input_parser)
te_data = te_data.map(input_parser)

#Create iterator object.
# tf iterator object
#we can iteratate over data. params refer to data type of image, and dimensionality 
iterator = Iterator.from_structure(tr_data.output_types, 
                                    tr_data.output_shapes)
                        
next_element = iterator.get_next()

# initialization
#make initializer with tr_data and te_data
#necessary for tensorflow to make connection between real data and variables called placeholders which data flows into.
#placeholders are the the data structure that serve as inputs.
training_init_op = iterator.make_initializer(tr_data)
testing_init_op = iterator.make_initializer(te_data)

with tf.Session() as sess:
    #initialize the iterator on the training data
    sess.run(training_init_op)

    #get each element of the training dataset until the end if reached
    while True:
        try:
            #looks at next element, gets item in list
            elem = sess.run(next_element)
            #add reading process of the image
            #y = Wx + b
            print(elem)
        except tf.errors.OutOfRangeError:
            print("End of training dataset")
            break

    #initialize the iterator on test data
    sess.run(testing_init_op)

    while True:
        try:
            elem = sess.run(next_element)
            print(elem)
        except tf.errors.OutOfRangeError:
            print("End of testing dataset")
            break

    #using queues for data

