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
NUM_ITERATIONS=$(python2 ${para_py} -p ${para_file} export_iteration_num)
# read batch size for inference
inf_batch_size=$(python2 ${para_py} -p ${para_file} inf_batch_size)
expr_name=$(python2 ${para_py} -p ${para_file} expr_name)

trail=iter${NUM_ITERATIONS}
frozen_graph=frozen_inference_graph_${trail}.pb

inf_dir=inf_results

SECONDS=0

#  remove the old results and inference
if [ -d "$inf_dir" ]; then
    rm -r $inf_dir
fi
python ${eo_dir}/grss_data_fusion/deeplab_inference.py --frozen_graph=${frozen_graph} --inf_output_dir=${inf_dir} --inf_batch_size=${inf_batch_size}

duration=$SECONDS
echo "$(date): time cost of inference: ${duration} seconds">>"time_cost.txt"




