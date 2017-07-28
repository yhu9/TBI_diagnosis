#Ynsheng Hu
#July 26, 2017
#Traumatic Brain Injury Study
#University of Alaska Anchorage Computer Science Department

#This module reads in the record of 3-d brain images created as csv files and resizes them
#to the correct output size. Three functions are implemented here.

#1. plotRecord(filename)
#2. readRecord(filename)
#3. readFiles(file_directory)

############################################################################################


#Basic imports for manipulation of the input data
import numpy as np
import os
import random
import constants
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D
from mayavi import mlab


#Create the container from the constants
CONTAINER = np.zeros((constants.CONTAINER_SIZE,constants.CONTAINER_SIZE,constants.CONTAINER_SIZE),np.uint16)

def plotRecord(filename):
    import matplotlib.pyplot as plt
    csv = np.genfromtxt(filename,delimiter=",")
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(csv[:,0],csv[:,1],csv[:,2])
    plt.show()

def readRecord(filename,flag):
    #(Optional) you can plot the record
    if flag == "show":
        plotRecord(filename)

    #Create a record and label to fill
    record = np.zeros((constants.OUTPUT_SIZE,constants.OUTPUT_SIZE,constants.OUTPUT_SIZE))
    label = 1

    #read the data
    data_in = np.genfromtxt(filename,delimiter=',')
    
    #read the label
    label = random.randint(0,1)
    
    #find the min value of the data_in
    min_val = np.amin(data_in,axis = 0)
    
    #shift to positive values >= 1
    data_in = data_in - min_val + 1
   
    #scale all values to integers between 0 - (constants.CONTAINER_SIZE - 1) using the max value across all
    max_val = np.amax(data_in)
    scale = max_val / (constants.CONTAINER_SIZE - 1)
    data_in = data_in / scale
    
    #turn everything into uint16 by taking the floor
    data_in = np.floor(data_in).astype(np.uint16)
    
    #populate container using the processed data
    for vals in data_in:
        CONTAINER[vals[0]][vals[1]][vals[2]] = CONTAINER[vals[0]][vals[1]][vals[2]] + 1

    #Resize the container into something smaller and add up anything that map into the same spot
    mult_val = constants.CONTAINER_SIZE / constants.OUTPUT_SIZE
    for i in range(constants.OUTPUT_SIZE):
        for j in range(constants.OUTPUT_SIZE):
            for k in range(constants.OUTPUT_SIZE):
                x = np.array(np.arange(i*mult_val,(i+1) * mult_val - 1),np.intp)
                y = np.array(np.arange(j*mult_val,(j+1) * mult_val - 1),np.intp)
                z = np.array(np.arange(k*mult_val,(k+1) * mult_val - 1),np.intp)
                val = np.sum(CONTAINER[np.ix_(x,y,z)])
                record[i][j][k] = val
 
    #Optional showing of the image before it is blurred
    if(flag == "show"):
        x,y,z = np.where(record >= 0)
        value = record.flatten()
        mlab.points3d(x,y,z,value)
        mlab.show()

    #Blur the record
    blur_kernal = np.zeros((3,3,3),np.float16)
    blur_kernal.fill(1)
    blur_kernal = blur_kernal * constants.BLUR_COEF
    ndimage.convolve(record,blur_kernal,mode='constant',cval=0.0)

    #Optional showing of the image after it is blurred
    if(flag == "show"):
        x,y,z = np.where(record >= 0)
        value = record.flatten()
        mlab.points3d(x,y,z,value)
        mlab.show()

    return record.flatten().astype(np.int32),label
   
#Read all the files to create the training data, validation data, training labels, and validation labels
#   INPUT
#
#1. directory of the images to be used
#
#   OUTPUT
#
#1. Object containing the training and evaluation data and labels
#
#########################################################################3
def readFiles(directory):
    #Get all files from the directory
    directory_list = os.listdir(directory)

    #Create containers for the data as numpy arrays
    train_images = np.zeros((constants.N,constants.OUTPUT_SIZE * constants.OUTPUT_SIZE * constants.OUTPUT_SIZE),dtype=np.float32)
    train_labels = np.zeros(constants.N,dtype=np.int64)
    eval_images = train_images
    eval_labels = train_labels

    for i,f in enumerate(directory_list):
        print("Processing: " + directory + f)
        train_images[i],train_labels[i] = readRecord(directory + f,"noshow")

    class INPUT(object):
        pass

    IN_DATA = INPUT()
    IN_DATA.train_data = train_images
    IN_DATA.train_labels = train_labels
    IN_DATA.eval_data = train_images
    IN_DATA.eval_labels = train_labels

    print("Done")

    return IN_DATA

#Simple function to show the record you want to see using the readRecord function and its flag
def showRecord(filepath):
    readRecord(filepath,"show")

#A dummy record for testing purposes
def dummyRecord():

    class INPUT(object):
        pass

    dummy_data_container = np.ones((constants.N,constants.OUTPUT_SIZE * constants.OUTPUT_SIZE * constants.OUTPUT_SIZE),dtype=np.int32)
    dummy_label_container = np.ones(constants.N,dtype=np.int64)

    dummy_data = np.ones((constants.OUTPUT_SIZE * constants.OUTPUT_SIZE * constants.OUTPUT_SIZE),dtype=np.int32)
    dummy_label = 0
    
    dummy_data_container[0] = dummy_data
    dummy_label_container[0] = dummy_label

    IN_DATA = INPUT()
    IN_DATA.train_data = dummy_data_container
    IN_DATA.train_labels = dummy_label_container
    IN_DATA.eval_data = dummy_data_container
    IN_DATA.eval_labels = dummy_label_container

    print("Done")



    return IN_DATA

