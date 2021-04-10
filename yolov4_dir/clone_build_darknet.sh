#!/bin/bash

# clone and build darkent

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 27 March, 2021

# need singularity >= 3.6
# run this inside the singularity container (with CUDA and cudnn).

# install  darknet
    cd ~/programs && \
    git clone https://github.com/AlexeyAB/darknet
    # go into the folder, change makefile to have GPU and OPENCV enabled
    cd darknet
    # use the verion latest commit on March 28, 2021
    git reset --hard 1e3a616ed6cefc517db6c8c106c83de24fad275c

    sed -i 's/OPENCV=0/OPENCV=1/' Makefile \
    && sed -i 's/GPU=0/GPU=1/' Makefile && sed -i 's/CUDNN=0/CUDNN=1/' Makefile \
    && sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile \
    && sed -i 's/LIBSO=0/LIBSO=1/' Makefile \
    && make