#!/bin/bash

# run the script inside a singularity container.
# before running this script, need to set environment using 'env_setting.sh'

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 11 November, 2020

# run data preparing, training, inference, and post processing
exe_script=./download_planet_northern_alaska_2020_July_Aug.sh

sing_dir=${HOME}
sing_img=${sing_dir}/programs/ubuntu20.04_curc.simg

#env_home=${sing_dir}/packages
env_home=${sing_dir}

# some issues related to ENV variable setting on ITSC services, but is ok on Cryo03.
# https://github.com/sylabs/singularity/issues/3510

# HOME will be overwrite by the value in host machine, no matter what I did, it still be overwritten.
# LD_LIBRARY_PATH has been overwrite as well, but if we unset LD_LIBRARY_PATH or use --cleanenv, then the problem is solved


# set environment 
SINGULARITYENV_HOME=${HOME} \
SINGULARITYENV_TZ=America/Denver \
SINGULARITYENV_PATH=/bin:${env_home}/bin:${env_home}/programs/miniconda3/bin:$PATH \
SINGULARITYENV_GDAL_DATA=${env_home}/programs/miniconda3/share/gdal \
SINGULARITYENV_LD_LIBRARY_PATH=${env_home}/programs/miniconda2/lib:$LD_LIBRARY_PATH \
singularity exec ${sing_img} ${exe_script}





