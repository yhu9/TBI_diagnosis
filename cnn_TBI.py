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

# Flatten Functin for multiple convolution kernels in a single layer
def flatten_function(tensor_in):
    tensor_in_shape = tensor_in.get_shape()
    tensor_in_flat = tf.reshape(tensor_in,[tensor_in_shape[0].value or -1, np.prod(tensor_in_shape[1:]).value])
    return tensor_in_flat

# Model creation function
def cnn_model_fn(features,labels,mode,params):
    #merge all the summaries and write them out to
    # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
    
    #Input Layer
    #4-d tensor: [batch_size,image_depth,image_width,image_height,channels]
    #TBI images are 25x25x25 pixels, 1-channel 
    #a -1 batch_size indicates a dynamic allocation based on feature size
    input_layer = tf.reshape(features, [-1,constants.IMAGE_SIZE, constants.IMAGE_SIZE,constants.IMAGE_SIZE,constants.IMAGE_CHANNELS])

    #Convolutional Layer #1
    #computes 32 features using 3x3x3 filter with ReLU activation
    #Padding is added to preserve width and height
    #Input Tensor Shape: [batch_size,constants.IMAGE_SIZE,constants.IMAGE_SIZE,constants.IMAGE_DEPTH]
    #Output Tensor Shape: [batch_size,constants.IMAGE_SIZE,constants.IMAGE_SIZE,32]
    conv1 = tf.layers.conv3d(
            inputs=input_layer,
            filters=constants.N_FEAT_LAYER1,
            kernel_size=[3,3,3],
            padding="SAME",
            activation=tf.nn.relu
            )

    #Pooling Layer #1
    #First max pooling layer with a 2x2x2 filter and stride of 2
    #Input tensor shape: [batch_size,25,25,25,32]
    #Output tensor shape: [batch_size,12,12,12,32]
    pool1 = tf.layers.max_pooling3d(
            inputs=conv1,
            pool_size=[2,2,2],
            strides=2)

    #Convlutional Layer #2
    #Computes 64 features using a 5x5 filter
    #Padding is used to preserve width and height
    #Input tensor Shape: [batch_size, 12,12,12,32]
    #Output tensor shape: [batch_size,12,12,12,64]
    conv2 = tf.layers.conv3d(
            inputs=pool1,
            filters=constants.N_FEAT_LAYER2,
            kernel_size=[3,3,3],
            padding="same",
            activation=tf.nn.relu)
    
    #Pooling Layer #2
    #second max pooling layer with 2x2 filter and stride of 2
    #Input tensor shape: [batch_size, 12,12,12,64]
    #Output tensor shape:[batch_size,6,6,6,64]
    pool2 = tf.layers.max_pooling3d(
            inputs=conv2,
            pool_size=[2,2,2],
            strides=2)

    #Flatten Tensor into a batch of vectors
    pool2_flat = tf.reshape(pool2,[-1,6*6*6* constants.N_FEAT_LAYER2])

    #Fully Connected Layer (aka dense layer)
    #Dense Layer with 1024 neurons
    #Input Tensor Shape:[batch_size,6*6*6*64]
    #Output Tensor shape:[batch_size,6*6*6*64]
    dense = tf.layers.dense(
            inputs=pool2_flat,
            units=1024,
            activation=tf.nn.relu)

    #Dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
            inputs=dense,
            rate=0.4,
            training=mode == learn.ModeKeys.TRAIN)

    #Logits Layer
    #Input Tensor shape: [batch_size,6*6*6*64]
    #Output Tensor Shape: [batch_size,2]
    output_layer = tf.layers.dense(inputs=dropout,units=constants.N_CLASSES)

    #Calculate Loss
    loss = None

    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels,tf.int32), depth=constants.N_CLASSES)
        loss = tf.losses.softmax_cross_entropy(
                onehot_labels = onehot_labels,
                logits=output_layer)
        tf.summary.scalar('loss',loss)


    #Configure the Training Operation
    train_op = None
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
                loss = loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=params["learning_rate"],
                optimizer=params["optimizer"])

    #Generate Predictions
    predictions = tf.reshape(output_layer,[-1])
    predictions_dict = {
            "classes":tf.argmax(input=output_layer, axis=1),
            "probabilities":tf.nn.softmax(output_layer,name="softmax_tensor")
            }
    
    #Eval Metric
    eval_metric_ops = {
            "rmse" : tf.metrics.root_mean_squared_error(
                    tf.cast(labels,tf.float32),predictions)
            }

    #Merge all summaries and write them for tensorboard visualization
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(constants.LOG_DIR)
    
    #Write summaries every n steps
    #for i in range(constants.STEPS):
    #    summary, _ = sess.run([merged, train_op], feed_dict=feed_dict(True))
    #    train_writer.add_summary(summary, i)
        
    train_writer.close()
    
    #Return the generated model
    return model_fn_lib.ModelFnOps(
            mode=mode,
            predictions=predictions_dict,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops)

#######################################################################################
#Read input
def read_input():

    class INPUT(object):
        pass

    IN_DATA = INPUT()

    # Load training and eval data


    mnist = learn.datasets.load_dataset("mnist")
    IN_DATA.train_data = mnist.train.images
    IN_DATA.train_labels = np.asarray(mnist.train.labels,dtype=np.int32) #Returns np.array
    IN_DATA.eval_data = mnist.test.images
    IN_DATA.eval_labels = np.asarray(mnist.test.labels,dtype=np.int32) #Returns np arra
    IN_DATA.validation = mnist.validation
    
    return IN_DATA

#######################################################################################
#Main Function

def main(unused_argv):
    #train/eval data are represented as 2-d numpy array
    #train/eval labels are represented as a 1-d numpy array
    #First lets read in all of the data all located in the directory
    input_data = filereader.readFiles(constants.DIRECTORY)

    #set model params
    model_params = {"learning_rate": constants.LEARNING_RATE, "optimizer":constants.OPTIMIZER}

    #create the estimator
    mnist_classifier = tf.contrib.learn.Estimator(model_fn=cnn_model_fn, model_dir=constants.LOG_DIR,params=model_params)
    
    #Train the model
    mnist_classifier.fit(
            x=input_data.train_data,
            y=input_data.train_labels,
            batch_size=constants.BATCH_SIZE,
            steps=constants.STEPS
            #monitors=[learn.monitors.ValidationMonitor(input_data.validation.images,input_data.validation.labels)]
            )

    #Configure the accuracy metric for evaluation
    validation_metrics = {
        "accuracy":
            tf.learn.MetricSpec(
                metric_fn=tf.metrics.streaming_accuracy,
                prediction_key=tf.learn.PredictionKey.CLASSES),
        "precision":
            tf.learn.MetricSpec(
                metric_fn=tf.metrics.streaming_precision,
                prediction_key=tf.learn.PredictionKey.CLASSES),
        "recall":
            tf.learn.MetricSpec(
                metric_fn=tf.metrics.streaming_recall,
                prediction_key=tf.learn.PredictionKey.CLASSES)
    }

    #Evaluate the model and get the results printed out
    eval_results = mnist_classifier.evaluate(
            x=eval_data, y=eval_labels, metrics=validation_metrics)
    print(eval_results)

if __name__ == "__main__":
    tf.app.run()

