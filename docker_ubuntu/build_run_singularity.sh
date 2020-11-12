#!/bin/bash

# singularity for running tensorflow on Centos (ITSC services)
# This container only provides a OS (ubuntu 16.04) for running our scirpt
# It will mount ~/codes, ~/programs, ~/Data: mount $HOME as /home/hlc
# The python, tensorflow, and other depends are in ~/programs, we will copy them from Cryo03
# Python is ~/programs/anaconda2/bin/  (Deeplabv3+ need python2)

#build singularity docker
# https://github.com/NIH-HPC/Singularity-Tutorial/tree/master/01-building

# --sandbox option in the command above tells Singularity that we want to
# build a special type of container for development purposes. Then, ubuntu16.04_itsc_tf.img is a folder
# the default is to build a squashfs image, it a file
# build command need "sudo"
sudo singularity build --sandbox ubuntu16.04_itsc_tf.img ubuntu16.4.recipe
# build to squashfs format, cannot modified insides
sudo singularity build ubuntu16.04_itsc_tf.simg ubuntu16.4.recipe

sudo singularity build ubuntu20.04_itsc_tf.simg ubuntu20.4.recipe

# Using shell to explore and modify containers
# it is nice that the container have the same user and home folder on the host machine
# but it doesn't source ".bashrc". The linked folders also are invalid
singularity shell ubuntu16.04_itsc_tf.img
singularity shell ubuntu16.04_itsc_tf.simg

# it seem every time, I have to set LD_LIBRARY_PATH before import tensorflow in python, or it complains
# cannot load libcudnn.so.7, but I have set the LD_LIBRARY_PATH in %environment
# add --nv (after shell) to support nvidia
singularity shell --nv  ubuntu16.04_itsc_tf.simg

# IN the container, use "hostname" to show the name of the host machine
# run a script inside singularity container
singularity exec ubuntu16.04_itsc_tf.simg hostname

singularity exec ubuntu16.04_itsc_tf.simg python

# mount folder
# The --bind/-B option can be specified multiple times,
# or a comma-delimited string of bind path specifications can be used.

#cryo03:
export SINGULARITY_BINDPATH=/500G:/500G,/DATA1:/DATA1
# or
singularity exec --bind /500G:/500G,/DATA1:/DATA1 ubuntu16.04_itsc_tf.simg ls -l /DATA1













