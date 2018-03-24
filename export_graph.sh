#!/usr/bin/env bash

trail=$1

para_file=para.ini
para_py=/home/hlc/codes/PycharmProjects/DeeplabforRS/parameters.py
WORK_DIR=$(pwd)
tf_research_dir="/home/hlc/codes/PycharmProjects/tensorflow/models/research"
deeplab_dir=${tf_research_dir}/deeplab

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:${tf_research_dir}
export PYTHONPATH=$PYTHONPATH:${tf_research_dir}/slim

# Set up the working directories.
expr_name=$(python2 ${para_py} -p ${para_file} expr_name)
EXP_FOLDER=${expr_name}

TRAIN_LOGDIR="${WORK_DIR}/${EXP_FOLDER}/train"
EXPORT_DIR="${WORK_DIR}/${EXP_FOLDER}/export"

batch_size=$(python2 ${para_py} -p ${para_file} batch_size)
iteration_num=$(python2 ${para_py} -p ${para_file} iteration_num)
base_learning_rate=$(python2 ${para_py} -p ${para_file} base_learning_rate)

output_stride=$(python2 ${para_py} -p ${para_file} output_stride)
atrous_rates1=$(python2 ${para_py} -p ${para_file} atrous_rates1)
atrous_rates2=$(python2 ${para_py} -p ${para_file} atrous_rates2)
atrous_rates3=$(python2 ${para_py} -p ${para_file} atrous_rates3)

NUM_ITERATIONS=${iteration_num}

# Export the trained checkpoint.
CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph_${trail}.pb"

python "${deeplab_dir}"/export_model.py \
  --logtostderr \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_PATH}" \
  --model_variant="xception_65" \
  --atrous_rates=12 \
  --atrous_rates=24 \
  --atrous_rates=36 \
  --output_stride=8 \
  --decoder_output_stride=4 \
  --num_classes=21 \
  --crop_size=513 \
  --crop_size=513 \
  --inference_scales=1.0