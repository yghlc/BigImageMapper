#!/bin/bash

# Compile the Darknet C program.
#

#First Set GPU=1 CUDNN=1, CUDNN_HALF=1, OPENCV=1 in /yoltv4/darknet/Makefile, then make:

# run this iin containder

dir=~/codes/PycharmProjects/yghlc_yoltv4

cd ${dir}/darknet
make