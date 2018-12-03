#!/usr/bin/env bash

echo $(basename $0) : "remove previous data or results to run again"
#introduction: remove previous data or results to run again
#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 28 October, 2018


# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

para_file=$1
if [ ! -f $para_file ]; then
   echo "File ${para_file} not exists in current folder: ${PWD}"
   exit 1
fi

deeplabRS=~/codes/PycharmProjects/DeeplabforRS
para_py=${deeplabRS}/parameters.py


# remove sub-images
subImage_dir=$(python2 ${para_py} -p ${para_file} input_train_dir)
subLabel_dir=$(python2 ${para_py} -p ${para_file} input_label_dir)

if [ -d "${subImage_dir}" ]; then
    rm -r ${subImage_dir}
fi
if [ -d "${subLabel_dir}" ]; then
    rm -r ${subLabel_dir}
fi
