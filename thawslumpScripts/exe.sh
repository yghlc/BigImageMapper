#!/bin/bash

eo_dir=/home/hlc/codes/PycharmProjects/Landuse_DL
cd ${eo_dir}
git pull
cd -

expr=${PWD}
SECONDS=0

#set GPU on Cryo06
export CUDA_VISIBLE_DEVICES=1

## split images
#${eo_dir}/preparing_traindata.sh
##${eo_dir}/training_img_augment.sh

#exit

## convert to TFrecord 
#python ${eo_dir}/datasets/build_muti_lidar_data.py

# training
${eo_dir}/deeplab_mutiLidar_train.sh


duration=$SECONDS
echo "$(date): time cost: ${duration} seconds">>"time_cost.txt"

