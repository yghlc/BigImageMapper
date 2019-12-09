#!/bin/bash

# run the script inside a singularity container.
# before running this script, need to set environment using 'env_setting.sh'

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 9 December, 2019

# run data preparing, training, inference, and post processing
exe_script=./exe.sh

sing_dir=${HOME}
sing_img=${sing_dir}/ubuntu16.04_itsc_tf.simg

env_home=${sing_dir}/packages

# set environment 
SINGULARITYENV_HOME=${sing_dir}/packages \
SINGULARITYENV_TZ=Asia/Hong_Kong \
SINGULARITYENV_PATH=/bin:${env_home}/bin:${env_home}/programs/miniconda2/bin:$PATH \
SINGULARITYENV_GDAL_DATA=${env_home}/programs/miniconda2/share/gdal \
SINGULARITYENV_LD_LIBRARY_PATH=${env_home}/programs/cuda-9.0/lib64:${env_home}/programs/cuDNN_7.0/cuda/lib64:${env_home}/programs/miniconda2/lib:$LD_LIBRARY_PATH \
singularity exec --nv ${sing_img} ${exe_script}





