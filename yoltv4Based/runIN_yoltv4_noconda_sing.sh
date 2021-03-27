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

#export SINGULARITY_BINDPATH=/500G:/500G,/DATA1:/DATA1,/home/hlc:/home/hlc

#SINGULARITYENV_LD_LIBRARY_PATH=/usr/lib64:${yoltv4}/lib:${LD_LIBRARY_PATH} \

# becasue SINGULARITYENV_PATH will overwrite some env inside the container, so we set them manually
PATH_IN_SINGULARITY=/usr/local/cuda/bin:/usr/sbin:/usr/bin:/sbin:/bin
#LIBRARY_PATH_IN_SINGULARITY=

# set environment
SINGULARITYENV_TZ=America/Denver \
SINGULARITYENV_PATH=${yoltv4}/bin:${PATH_IN_SINGULARITY} \
SINGULARITYENV_GDAL_DATA=${yoltv4}/share/gdal \
singularity exec --nv ${sing_img} ${exe_script}