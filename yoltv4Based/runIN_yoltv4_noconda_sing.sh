#!/bin/bash

# run the script inside a singularity container.

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 27 March, 2021

# need singularity >= 3.6

# running script
exe_script=./exe.sh


sing_dir=${HOME}
sing_img=${sing_dir}/programs/yoltv4_noconda.sif


yoltv4=~/programs/miniconda3/envs/yoltv4

# more about the environment setting: https://sylabs.io/guides/3.7/user-guide/environment_and_metadata.html

#export SINGULARITY_BINDPATH=/500G:/500G,/DATA1:/DATA1,/home/hlc:/home/hlc
#SINGULARITYENV_LD_LIBRARY_PATH=/usr/lib64:${yoltv4}/lib:${LD_LIBRARY_PATH} \

# use SINGULARITYENV_APPEND_PATH or SINGULARITYENV_PREPEND_PATH to add path on the host machine
export SINGULARITYENV_PREPEND_PATH=${yoltv4}/bin


# set environment
SINGULARITYENV_TZ=America/Denver \
SINGULARITYENV_GDAL_DATA=${yoltv4}/share/gdal \
singularity exec --nv ${sing_img} ${exe_script}