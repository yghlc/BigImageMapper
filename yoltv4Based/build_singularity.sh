#!/bin/bash

# build singularity container for yoltv4

#  build a sanbox, for testing

#sudo singularity build --sandbox yoltv4_sanbox yoltv4_2.def
# get error: 2021/03/27 13:55:33  info unpack layer: sha256:4007a89234b4f56c03e6831dc220550d2e5fba935d9f5f5bcea64857ac4f4888
#FATAL:   While performing build: packer failed to pack: while unpacking tmpfs: error unpacking rootfs: unpack layer: unpack entry: bin/uncompress:
# link: link rootfs-0f87c549-8f04-11eb-a113-0800272be0ec/bin/gunzip rootfs-0f87c549-8f04-11eb-a113-0800272be0ec/bin/uncompress:
# operation not permitted

# This error keep occuring when build container on Vagrant, even change singularity from 3.5 to 3.6.
# but when building container on curc (ssh scompile, module load singularity**), don't have this error.

# try build sandbox one by one
#singularity build --sandbox yoltv4_sanbox  docker://nvidia/cuda:9.2-devel-ubuntu16.04


###############################################################
#singularity build  yoltv4.sif yoltv4_2.def

sudo singularity build --sandbox yoltv4_noconda yoltv4_noConda.def

sudo singularity build  yoltv4_noconda.sif yoltv4_noConda.def


# Using shell to explore containers
# it is nice that the container have the same user and home folder on the host machine
# but it doesn't source ".bashrc". The linked folders also are invalid
singularity shell yoltv4_noconda.sif

# it seem every time, I have to set LD_LIBRARY_PATH before import tensorflow in python, or it complains
# cannot load libcudnn.so.7, but I have set the LD_LIBRARY_PATH in %environment
# add --nv (after shell) to support nvidia
singularity shell --nv yoltv4_noconda.sif

# IN the container, use "hostname" to show the name of the host machine
# run a script inside singularity container
singularity exec yoltv4_noconda.sif hostname

singularity exec yoltv4_noconda.sif python