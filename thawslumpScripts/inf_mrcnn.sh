#!/bin/bash

#introduction: perform inference
#
#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 26 October, 2018

# Exit immediately if a command exits with a non-zero status.
set -e

# input a parameter: the path of para_file (e.g., para.ini)
para_file=$1
if [ ! -f $para_file ]; then
   echo "File ${para_file} not exists in current folder: ${PWD}"
   exit 1
fi

eo_dir=~/codes/PycharmProjects/Landuse_DL

para_py=~/codes/PycharmProjects/DeeplabforRS/parameters.py

trained_model=$(python2 ${para_py} -p ${para_file} trained_model)
inf_dir=inf_results

SECONDS=0

#  remove the old results and inference
if [ -d "$inf_dir" ]; then
    rm -r $inf_dir
fi

~/programs/anaconda3/bin/python3 ${eo_dir}/thawslumpScripts/thawS_rs_maskrcnn.py inference_rsImg_multi \
    --para_file=${para_file} \
    --model=${trained_model}


duration=$SECONDS
echo "$(date): time cost of inference: ${duration} seconds">>"time_cost.txt"


