#!/usr/bin/env bash

#introduction: Run prediction using trained model of DL, only inference one images
# only run on Cryo03 or ITSC service
#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 24 Sep, 2019


##MAKE SURE the /usr/bin/python, which is python2 on Cryo06
#export PATH=/usr/bin:$PATH
# python2 on Cryo03, tensorflow 1.6
export PATH=~/programs/anaconda2/bin:$PATH

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace


eo_dir=~/codes/PycharmProjects/Landuse_DL

para_file=$1
para_py=~/codes/PycharmProjects/DeeplabforRS/parameters.py


inf_dir=$2
inf_list_file=$3
export CUDA_VISIBLE_DEVICES=$4
################################################

expr_name=$(python2 ${para_py} -p ${para_file} expr_name)

NUM_ITERATIONS=$(python2 ${para_py} -p ${para_file} export_iteration_num)
trail=iter${NUM_ITERATIONS}
frozen_graph=frozen_inference_graph_${trail}.pb


SECONDS=0
################################################
## inference and post processing, including output "time_cost.txt"
## each time,only inference one image
inf_batch_size=$(python2 ${para_py} -p ${para_file} inf_batch_size)
python ${eo_dir}/grss_data_fusion/deeplab_inference.py --frozen_graph=${frozen_graph} \
--inf_output_dir=${inf_dir} --inf_batch_size=${inf_batch_size} --inf_list_file=${inf_list_file} \
--inf_para_file=${para_file}


duration=$SECONDS
echo "$(date): time cost of inference for image in ${inf_list_file}: ${duration} seconds">>"time_cost.txt"

# write a file to indicate that the prediction has done.
echo ${inf_list_file} > ${inf_list_file}+'_done'

