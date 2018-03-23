#!/usr/bin/env bash


# try to repeat the author's result
# train from the pre-trained model only use ImageNet
# I only have a samll GPU, so it is hard to repeat the author's,
# but I hope have close result and watch the training process
# modified from "tensorflow/models/research/deeplab/local_test.sh"

para_file=para.ini
para_py=/home/hlc/codes/PycharmProjects/DeeplabforRS/parameters.py

tf_research_dir="/home/hlc/codes/PycharmProjects/tensorflow/models/research"
# test dir:"/home/hlc/Data/2018_IEEE_GRSS_Data_Fusion/deeplabv4_2_lidar",
# go to the folder first
WORK_DIR=$(pwd)
deeplab_dir=${tf_research_dir}/deeplab

# go to work dir
#cd $WORK_DIR

#
# This script is used to run local test on PASCAL VOC 2012. Users could also
# modify from this script for their use case.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./local_test.sh
#
#

# Exit immediately if a command exits with a non-zero status.
set -e


# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:${tf_research_dir}
export PYTHONPATH=$PYTHONPATH:${tf_research_dir}/slim


# Set up the working environment.
#CURRENT_DIR=$(pwd)
#WORK_DIR="${CURRENT_DIR}/deeplab"

# Run model_test first to make sure the PYTHONPATH is correctly set.
#python "${WORK_DIR}"/model_test.py -v

# Set up the working directories.
expr_name=$(python2 ${para_py} -p ${para_file} expr_name)
EXP_FOLDER=${expr_name}

INIT_FOLDER="${WORK_DIR}/${EXP_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${EXP_FOLDER}/export"

mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

# Copy locally the trained checkpoint as the initial checkpoint.
TF_INIT_CKPT="deeplabv3_xception_2018_01_04.tar.gz"     #xception

cd "${INIT_FOLDER}"
cp ~/Data/deeplab/v3+/pre-trained_model/${TF_INIT_CKPT} .
tar -xf "${TF_INIT_CKPT}"

cd "${WORK_DIR}"

DATASET="${WORK_DIR}/tfrecord"

batch_size=$(python2 ${para_py} -p ${para_file} batch_size)
iteration_num=$(python2 ${para_py} -p ${para_file} iteration_num)
base_learning_rate=$(python2 ${para_py} -p ${para_file} base_learning_rate)

output_stride=$(python2 ${para_py} -p ${para_file} output_stride)
atrous_rates1=$(python2 ${para_py} -p ${para_file} atrous_rates1)
atrous_rates2=$(python2 ${para_py} -p ${para_file} atrous_rates2)
atrous_rates3=$(python2 ${para_py} -p ${para_file} atrous_rates3)

# Train 10 iterations.
NUM_ITERATIONS=${iteration_num}
python "${deeplab_dir}"/train.py \
  --logtostderr \
  --train_split="trainval" \
  --base_learning_rate=${base_learning_rate} \
  --model_variant="xception_65" \
  --atrous_rates=${atrous_rates1} \
  --atrous_rates=${atrous_rates2} \
  --atrous_rates=${atrous_rates3} \
  --output_stride=${output_stride} \
  --decoder_output_stride=4 \
  --train_crop_size=513 \
  --train_crop_size=513 \
  --train_batch_size=${batch_size} \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=False \
  --tf_initial_checkpoint="${INIT_FOLDER}/xception/model.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${DATASET}"

# Run evaluation. This performs eval over the full val split (1449 images) and
# will take a while.
# Using the provided checkpoint, one should expect mIOU=82.20%. (can read this on TensorBoard)
python "${deeplab_dir}"/eval.py \
  --logtostderr \
  --eval_split="val" \
  --model_variant="xception_65" \
  --atrous_rates=${atrous_rates1} \
  --atrous_rates=${atrous_rates2} \
  --atrous_rates=${atrous_rates3} \
  --output_stride=${output_stride} \
  --decoder_output_stride=4 \
  --eval_crop_size=513 \
  --eval_crop_size=513 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${DATASET}" \
  --max_number_of_evaluations=1

# Visualize the results.
#python "${deeplab_dir}"/vis.py \
#  --logtostderr \
#  --vis_split="val" \
#  --model_variant="xception_65" \
#  --atrous_rates=6 \
#  --atrous_rates=12 \
#  --atrous_rates=18 \
#  --output_stride=16 \
#  --decoder_output_stride=4 \
#  --vis_crop_size=513 \
#  --vis_crop_size=513 \
#  --checkpoint_dir="${TRAIN_LOGDIR}" \
#  --vis_logdir="${VIS_LOGDIR}" \
#  --dataset_dir="${DATASET}" \
#  --max_number_of_iterations=1

# Export the trained checkpoint.
CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"

python "${deeplab_dir}"/export_model.py \
  --logtostderr \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_PATH}" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --num_classes=21 \
  --crop_size=513 \
  --crop_size=513 \
  --inference_scales=1.0

# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.
