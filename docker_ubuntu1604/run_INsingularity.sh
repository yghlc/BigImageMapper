#!/bin/bash

# run exe.sh or other script with the environment of singularity
exe_script=exe.sh
sing_img=~/codes/PycharmProjects/Landuse_DL/docker_ubuntu1604/ubuntu16.04_itsc_tf.simg

# set mount disk on Cryo03
export SINGULARITY_BINDPATH=/500G:/500G,/DATA1:/DATA1

SINGULARITYENV_PATH=$PATH SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
singularity exec ${sing_img} ${exe_script}




