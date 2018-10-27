#!/usr/bin/env bash

#introduction: export the frozen inference graph (a pb file)
#
#authors: Huang Lingcao
#email:huanglingcao@gmail.com
# add time: March 2018
#modify time: 26 October, 2018

# input a parameter: the path of para_file (e.g., para.ini)
para_file=$1
if [ ! -f $para_file ]; then
   echo "File ${para_file} not exists in current folder: ${PWD}"
   exit 1
fi


para_py=~/codes/PycharmProjects/DeeplabforRS/parameters.py
WORK_DIR=$(pwd)
tf_research_dir=~/codes/PycharmProjects/tensorflow/models/research
deeplab_dir=${tf_research_dir}/deeplab

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:${tf_research_dir}
export PYTHONPATH=$PYTHONPATH:${tf_research_dir}/slim

# Set up the working directories.
expr_name=$(python2 ${para_py} -p ${para_file} expr_name)
EXP_FOLDER=${expr_name}

TRAIN_LOGDIR="${WORK_DIR}/${EXP_FOLDER}/train"
EXPORT_DIR="${WORK_DIR}/${EXP_FOLDER}/export"


output_stride=$(python2 ${para_py} -p ${para_file} output_stride)
atrous_rates1=$(python2 ${para_py} -p ${para_file} atrous_rates1)
atrous_rates2=$(python2 ${para_py} -p ${para_file} atrous_rates2)
atrous_rates3=$(python2 ${para_py} -p ${para_file} atrous_rates3)

# read the saved iteration_num from para.ini or assigned a value
NUM_ITERATIONS=$(python2 ${para_py} -p ${para_file} export_iteration_num)


trail=iter${NUM_ITERATIONS}

# Export the trained checkpoint.
CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph_${trail}.pb"

multi_scale=$(python2 ${para_py} -p ${para_file} export_multi_scale)

if [ "$multi_scale" -eq 1 ]; then
## multi-scale
python "${deeplab_dir}"/export_model.py \
  --logtostderr \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_PATH}" \
  --model_variant="xception_65" \
  --atrous_rates=${atrous_rates1} \
  --atrous_rates=${atrous_rates2} \
  --atrous_rates=${atrous_rates3} \
  --output_stride=${output_stride} \
  --decoder_output_stride=4 \
  --num_classes=21 \
  --crop_size=513 \
  --crop_size=513 \
  --inference_scales=0.5 \
   --inference_scales=0.75 \
   --inference_scales=1.0 \
   --inference_scales=1.25 \
   --inference_scales=1.5 \
   --inference_scales=1.75
elif [ "$multi_scale" -eq 0 ]; then
## single-scale
python "${deeplab_dir}"/export_model.py \
  --logtostderr \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_PATH}" \
  --model_variant="xception_65" \
  --atrous_rates=${atrous_rates1} \
  --atrous_rates=${atrous_rates2} \
  --atrous_rates=${atrous_rates3} \
  --output_stride=${output_stride} \
  --decoder_output_stride=4 \
  --num_classes=21 \
  --crop_size=513 \
  --crop_size=513 \
   --inference_scales=1.0

else
   echo "Wrong input of the multi_scale parameter"
   exit 1

fi