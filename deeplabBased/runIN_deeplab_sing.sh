#!/bin/bash

# run the script inside a singularity container.

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 22 June, 2021

# need singularity >= 3.6
# need to activate darknet for python scripts: conda activate darknet

# running script
exe_script=./exe.sh


sing_dir=${HOME}
sing_img=${sing_dir}/programs/ubuntu2004_cuda1000_cudnn74.sif



# more about the environment setting: https://sylabs.io/guides/3.7/user-guide/environment_and_metadata.html

#export SINGULARITY_BINDPATH=/500G:/500G,/DATA1:/DATA1,/home/hlc:/home/hlc
#export SINGULARITY_BINDPATH=/projects/lihu9680:/projects/lihu9680
#SINGULARITYENV_LD_LIBRARY_PATH=/usr/lib64:${yoltv4}/lib:${LD_LIBRARY_PATH} \

# use SINGULARITYENV_APPEND_PATH or SINGULARITYENV_PREPEND_PATH to add path on the host machine
#export SINGULARITYENV_PREPEND_PATH=${sing_dir}/programs/miniconda3/envs/darknet/bin:${sing_dir}/programs/darknet


# set environment
SINGULARITYENV_TZ=America/Denver \
singularity exec --nv ${sing_img} ${exe_script}
