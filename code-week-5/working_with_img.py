import tensorflow as tf

# we have two classes

NUM_CLASSES = 2

#important for reading process of image
def input_parser(img_path, label):
    # convert the label to one-hot encoding - we have two classes, so we have two "cables" [1 0] is spiral, [0 1] is twin.
    # vector representing what class it is important for outputting probabilities

    #label is data that want to transform
    #num class is depth (how many classes you have, size of vector for every single label)
    one_hot = tf.one_hot(label, NUM_CLASSES)

    #read images
    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_image(img_file, channels=3)

    return img_decoded, one_hot