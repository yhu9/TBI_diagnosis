#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for TBI, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import math
import filereader
import constants

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

###################################################################
    

###################################################################
#1. Convolutional layer
#2. Pooling layers
#3. Convolutional layer
#4. pooling layer
#5. Fully connected layer
#6. Logits layer
###################################################################
'''
#Run Configuration for recording the information
def create_run_config():
    config = tf.contrib.learn.RunConfig(
            save_summary_steps=constants.STEPS_RECORD
            )

    return config
'''
###################################################################

# Flatten Function for multiple convolution kernels in a single layer
def flatten_function(tensor_in):
    tensor_in_shape = tensor_in.get_shape()
    tensor_in_flat = tf.reshape(tensor_in,[tensor_in_shape[0].value or -1, np.prod(tensor_in_shape[1:]).value])
    return tensor_in_flat

#In order to get the next batch of random samples and labels we get some help from this function found online
#https://stackoverflow.com/questions/40994583/how-to-implement-tensorflows-next-batch-for-own-data
def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

#Some monitoring basics on the tensorflow website
#https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""

    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

#Function for creating a convolution layer with the summaries for visualization
def conv3d(x_in, W_in, strides_in, layer_name):
    return tf.nn.conv3d(x_in,W_in,strides=strides_in,padding='SAME',name=layer_name)

#Function for creating a pooling layer
def maxpool3d(x_in,ksize_in,strides_in,layer_name):
    return tf.nn.max_pool3d(x_in,ksize=ksize_in,strides=strides_in,padding='SAME',name=layer_name)

#Define our Convolutionary Neural Network from scratch
def CNN(x,y):
    with tf.name_scope('model'):
        #The magic number described in this tutorial
        #https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
        #It is calculated as follows:
        #
        #           ceil(x/strides1/strides2) * ceil(y/strides1/strides2) * ceil(z/strides1/strides2) * conv2_features
        #
        #This number represents the number of features that are to be fed into the first fully connected layer
        magic_number = int(math.ceil(constants.IMAGE_SIZE / 2 / 2) * math.ceil(constants.IMAGE_SIZE / 2 / 2) * math.ceil(constants.IMAGE_SIZE / 2 / 2) * constants.N_FEAT_LAYER2)

        #Define our initial weights and biases for each layer
                                                        #3x3x3 patches, 1 channel, N1 features to compute
        weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,3,1,constants.N_FEAT_LAYER1])),
                                                        #3x3x3 patches, N1 channels, N2 features to compute
                   'W_conv2':tf.Variable(tf.random_normal([3,3,3,constants.N_FEAT_LAYER1,constants.N_FEAT_LAYER2])),
                   'W_fc':tf.Variable(tf.random_normal([magic_number,constants.N_FEAT_FULL1])),
                   'out':tf.Variable(tf.random_normal([constants.N_FEAT_FULL1, constants.N_CLASSES]))}

        biases = {'b_conv1':tf.Variable(tf.random_normal([constants.N_FEAT_LAYER1])),
                   'b_conv2':tf.Variable(tf.random_normal([constants.N_FEAT_LAYER2])),
                   'b_fc':tf.Variable(tf.random_normal([constants.N_FEAT_FULL1])),
                   'out':tf.Variable(tf.random_normal([constants.N_CLASSES]))}
        
        #create our input layer
        #(input_tensor, shape=[batch_size, image_depth, image_width, image_height, 1]       #not sure what the 1 is at the end
        x = tf.reshape(x, shape=[-1,constants.IMAGE_SIZE,constants.IMAGE_SIZE,constants.IMAGE_SIZE,1],name="input_layer")
        
        #create our 1st convolutionary layer with histogram summaries of activations
        with tf.variable_scope('Convolution_1'):
            conv1 = conv3d(x_in=x,W_in=weights['W_conv1'],strides_in=[1,1,1,1,1],layer_name='conv_1')
            activations1 = tf.nn.relu(conv1 + biases['b_conv1'])
            variable_summaries(weights['W_conv1'])
            tf.summary.histogram('activations_1',activations1)
            print(conv1.name)

        #create our 1st pooling layer
        with tf.variable_scope('Pool_1'):
            pool1 = maxpool3d(x_in=activations1,ksize_in=[1,2,2,2,1],strides_in=[1,2,2,2,1],layer_name='pool1')

        #create our 2nd convolutionary layer with histogram summaries of activations
        with tf.variable_scope('Convolution_2'):
            conv2 = conv3d(x_in=pool1,W_in=weights['W_conv2'],strides_in=[1,1,1,1,1],layer_name='conv_2')
            activations2 = tf.nn.relu(conv2 + biases['b_conv2'])
            variable_summaries(weights['W_conv2'])
            tf.summary.histogram('activations_2',activations2)

        #create our 2nd pooling layer
        with tf.variable_scope('Pool_2'):
            pool2 = maxpool3d(x_in=activations2,ksize_in=[1,2,2,2,1],strides_in=[1,2,2,2,1],layer_name='pool2')

        #create our first fully connected layer
        with tf.variable_scope('Fully_Connected_1'):
            fullyConnected = tf.reshape(pool2,[-1,magic_number])
            activations3 = tf.nn.relu(tf.matmul(fullyConnected,weights['W_fc']) + biases['b_fc'])
            tf.summary.histogram('activations_3',activations3)

        #create our dropout layer
        with tf.variable_scope('Dropout'):
            dropout = tf.nn.dropout(activations3,constants.KEEP_RATE)

        #Final fully connected layer for classification
        with tf.variable_scope('Fully_Connected_2'):
            output = tf.matmul(dropout,weights['out'])+biases['out']

    #Record accuracy
    #onehot_labels = tf.one_hot(indices=tf.cast(y,tf.int64), depth=constants.N_CLASSES)
    #correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(onehot_labels, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #tf.summary.scalar('accuracy',accuracy)

    return output

#Training function for our neural network
def train_neural_network(x,y,t_data,t_labels,v_data,v_labels):

    #Make predictions and calculate loss and accuracy given the inputs and labels
    predictions = CNN(x,y)
    onehot_labels = tf.one_hot(indices=tf.cast(y,tf.int64), depth=constants.N_CLASSES)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions,labels=onehot_labels))
    optimizer = tf.train.GradientDescentOptimizer(constants.LEARNING_RATE).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(predictions,1),tf.argmax(onehot_labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.summary.scalar('accuracy',accuracy)

    #Dictionary for feeding in batches of data
    def feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train:
            xs, ys = next_batch(constants.BATCH_SIZE,t_data,t_labels)
        else:
            xs, ys = v_data, v_labels
        return {x: xs, y: ys}

    #Run the session/CNN and either train or record accuracies at given steps
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(constants.LOG_DIR + '/train',sess.graph)
        test_writer = tf.summary.FileWriter(constants.LOG_DIR + '/test')

        successful_runs = 0
        total_runs = 0

        for i in range(constants.STEPS):
            if i % constants.STEPS_RECORD == 0:
                summary, loss, acc = sess.run([merged,cost,accuracy],feed_dict=feed_dict(False))
                test_writer.add_summary(summary,i)
                print('step: ' + str(i) + '     ' +
                        'loss: ' + str(loss) + '     ' +
                        'accuracy: ' + str(acc))
            else:
                summary,_ = sess.run([merged,optimizer],feed_dict=feed_dict(True))
                train_writer.add_summary(summary,i)


        train_writer.close()

#######################################################################################
#######################################################################################
#Main Function

def main(unused_argv):
  
    graph = tf.Graph()
    with graph.as_default():
        #train/eval data are represented as 2-d numpy array
        #train/eval labels are represented as a 1-d numpy array
        #First lets read in all of the data all located in the directory
        #input_data = filereader.readFiles(constants.DIRECTORY)
        input_data = filereader.dummyRecord()
        x = tf.placeholder('float')
        y = tf.placeholder('float')
        
        train_neural_network(x,y,
                input_data.train_data,
                input_data.train_labels,
                input_data.eval_data,
                input_data.eval_labels)

if __name__ == "__main__":
    tf.app.run()

