#!/usr/bin/python

import cnn_TBI
import filereader
import sys

def fileCheck(file_name):
    try:
        open(file_name,'r')
        return 1
    except IOError:
        print "Error: file does not appear to exist"
        return 0

if len(sys.argv) == 2:
    if fileCheck(sys.argv[1]) == 1:
        filereader.showRecord(sys.argv[1])
else:
    print("wrong number of arguments")
    print("these number of arguments have not been implemented")
    
