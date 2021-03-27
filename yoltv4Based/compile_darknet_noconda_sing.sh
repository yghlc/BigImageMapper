#!/bin/bash

# run the script inside a singularity container.

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 27 March, 2021

# running
dir=~/codes/PycharmProjects/yghlc_yoltv4

cd ${dir}/darknet

sing_dir=${HOME}
sing_img=${sing_dir}/programs/yoltv4_noconda.sif

# when compile darknet, don't set use any variable outside

#SINGULARITYENV_TZ=America/Denver \
#SINGULARITYENV_PATH=/bin:${yoltv4}/bin:$PATH \
#SINGULARITYENV_LD_LIBRARY_PATH=/usr/lib64:${yoltv4}/lib:$LD_LIBRARY_PATH \
#SINGULARITYENV_GDAL_DATA=${yoltv4}/share/gdal \
singularity exec --nv ${sing_img} make