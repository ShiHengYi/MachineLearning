#!/usr/bin/env python
# coding: utf-8

# In[8]:


from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def load_data_padding():
    print("Start to load data set using tensorflow example.")
    mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
    #extract data from example data set
    image_train = mnist.train.images
    image_validation = mnist.validation.images
    image_test = mnist.test.images

    label_train = mnist.train.labels
    label_validation = mnist.validation.labels  
    label_test = mnist.test.labels
    #  Reshape Input: image padding from 28x28 to 32x32 with filling with 0's
    image_train      = np.pad(image_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    image_validation = np.pad(image_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    image_test       = np.pad(image_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    return image_train, image_validation, image_test, label_train, label_validation, label_test
     
# In[9]:


import tensorflow as tf
from tensorflow.contrib.layers import flatten

#i have noticed that if we choose a smaller stddev value, the accuracy will increase.
#so I choosed stddev to be 0.01 which will provide a 98.8% accuracy

def network_Model(image):

    #C1: The output of C1 should be 28 · 28 · 6
    cw_1 = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=0, stddev=0.01))
    cb_1 = tf.Variable(tf.zeros(shape=6, dtype=tf.float32, name="cbias1"))
    c_1  = tf.nn.conv2d(input=image, filter=cw_1, strides=[1, 1, 1, 1], padding='VALID') + cb_1
    c_1  = tf.nn.relu(c_1, name="first_convolution")

    #S2: The output of S1 should be 14 · 14 · 6
    p_1 = tf.nn.max_pool(	value=c_1, 
    						ksize=[1, 2, 2, 1], 
    						strides=[1, 2, 2, 1], 
    						padding='VALID')
    
    #C3: The output of C3 should be 10 · 10 · 16
    cw_2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=0, stddev=0.01))
    cb_2 = tf.Variable(tf.zeros(shape=16, dtype=tf.float32, name="cbias2"))
    c_2  = tf.nn.conv2d(input=p_1, filter=cw_2, strides=[1, 1, 1, 1], padding='VALID') + cb_2
    c_2  = tf.nn.relu(c_2, name="second_convolution")

    #S4: The output of S4 should be 5 · 5 · 16
    p_2 = tf.nn.max_pool(	value=c_2, 
    						ksize=[1, 2, 2, 1], 
    						strides=[1, 2, 2, 1], 
    						padding='VALID')

    #Flatten: need to flat data before doing the fully connection.
    flat = flatten(p_2)
    
    #C5(F5): This layer should have 120 outputs
    fcw_1 = tf.Variable(tf.truncated_normal(shape=(400,120), mean=0, stddev=0.01))
    fcb_1 = tf.Variable(tf.zeros(shape=120, dtype=tf.float32, name="fbias1"))
    fc_1  = tf.matmul(flat, fcw_1) + fcb_1
    fc_1  = tf.nn.relu(fc_1, name="first_fully_connection")

    #F6: This layer should have 84 outputs:
    fcw_2 = tf.Variable(tf.truncated_normal(shape=(120,84), mean=0, stddev=0.01))
    fcb_2 = tf.Variable(tf.zeros(shape=84, dtype=tf.float32, name="fbias2"))
    fc_2  = tf.matmul(fc_1, fcw_2) + fcb_2
    fc_2  = tf.nn.relu(fc_2, name="second_fully_connection")

    #Output: This layer should have 10 outputs (Indicating 0 to 9).
    gcw = tf.Variable(tf.truncated_normal(shape=(84,10), mean=0, stddev=0.01))
    gcb = tf.Variable(tf.zeros(shape=10, dtype=tf.float32, name="gaussian_bias"))

    #Create Logits
    logits = tf.matmul(fc_2, gcw) + gcb
    return logits

# In[10]:

#read smoothed mnist data
image_train, image_validation, image_test, label_train, label_validation, label_test = load_data_padding()

EPOCHS = 10
BATCH_SIZE = 64
learning_rate = 0.001

#create place holder for both imputs
image_holder = tf.placeholder(  tf.float32, 
                                [None, 32, 32, 1], 
                                name="images")
label_holder = tf.placeholder(  tf.int32, 
                                [None],
                                name="labels")

#create 10x10 matrix for label holder and feed image place holders into modeling functions
logits = network_Model(image_holder)
label_matrix = tf.one_hot(label_holder, 10)

#create variable for training phase
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_matrix)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_operator = optimizer.minimize(loss_operation)

#create variable for testing phase
predictor = tf.equal(tf.argmax(logits, 1), tf.argmax(label_matrix, 1))
accuracy_check = tf.reduce_mean(tf.cast(predictor, tf.float32))
saver = tf.train.Saver()

#training and testing phase
with tf.Session() as sess:
    print("Initiating training process!")
    sess.run(tf.global_variables_initializer())
    for i in range(EPOCHS):
        print("Working on EPOCH {}!".format(i+1))
        for batch in range(0, len(image_train), BATCH_SIZE):
            sample_set = batch + BATCH_SIZE
            batch_images, batch_labels = image_train[batch:sample_set], label_train[batch:sample_set]
            sess.run(training_operator, feed_dict={image_holder: batch_images, label_holder: batch_labels})
        print("Done!")
    print("Training phase finished!")

    print("Initiating testing process!")
    accurate_counter = 0
    for batch in range(0, len(image_test), BATCH_SIZE):
        batch_images, batch_labels = image_test[batch:batch+BATCH_SIZE], label_test[batch:batch+BATCH_SIZE]
        accuracy_temp = sess.run(accuracy_check, feed_dict={image_holder: batch_images, label_holder: batch_labels})
        accurate_counter += (accuracy_temp * len(batch_images))
    accuracy = accurate_counter / len(image_test)
    print("Testing accuracy = {:.4f}".format(accuracy))