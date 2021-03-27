#!/bin/bash

# run the script inside a singularity container.

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 27 March, 2021

# running script
exe_script=./exe.sh


sing_dir=${HOME}
sing_img=${sing_dir}/programs/yoltv4_noconda.sif


yoltv4=~/programs/miniconda3/envs/yoltv4

# HOME will be overwrite by the value in host machine, no matter what I did, it still be overwritten.
# LD_LIBRARY_PATH has been overwrite as well, but if we unset LD_LIBRARY_PATH or use --cleanenv, then the problem is solved

#SINGULARITYENV_LD_LIBRARY_PATH=/usr/lib64:${env_home}/programs/cuda-10.0/lib64:${env_home}/programs/cuDNN_7.4_cuda10/cuda/lib64:/usr/lib64:$LD_LIBRARY_PATH \

#SINGULARITYENV_LD_LIBRARY_PATH=${env_home}/programs/miniconda3/lib:$LD_LIBRARY_PATH \
# set environment
SINGULARITYENV_TZ=America/Denver \
SINGULARITYENV_PATH=/bin:${yoltv4}/bin:$PATH \
SINGULARITYENV_LD_LIBRARY_PATH=/usr/lib64:${yoltv4}/lib:$LD_LIBRARY_PATH \
SINGULARITYENV_GDAL_DATA=${yoltv4}/share/gdal \
singularity exec --nv ${sing_img} ${exe_script}