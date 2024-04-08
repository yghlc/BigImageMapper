#!/usr/bin/env bash

#introduction: Run the whole process of fine-tuning Segment Anything model
#
#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 7 April, 2024

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

# set GPUs want to used (e.g. use GPU 0 & 1)
#export CUDA_VISIBLE_DEVICES=0,1

# the main parameter files
para_file=main_para.ini

# Landuse_DL scripts dir
eo_dir=~/codes/PycharmProjects/Landuse_DL

################################################
SECONDS=0
# remove previous data or results if necessary
#rm time_cost.txt || true
${eo_dir}/workflow/remove_previous_data.py ${para_file}

#extract sub_images based on the training polgyons
${eo_dir}/workflow/get_sub_images_multi_regions.py ${para_file}

################################################
## preparing training images.

${eo_dir}/workflow/split_sub_images.py ${para_file}
#${eo_dir}/workflow/training_img_augment.py ${para_file}
${eo_dir}/workflow/split_train_val.py ${para_file}

################################################
## run within conda environment (name: pytorch)
### training SAM
conda run --no-capture-output -n pytorch bash -c "${eo_dir}/sam_dir/fine_tune_sam.py ${para_file}"
################################################

