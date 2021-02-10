#!/usr/bin/env bash

#introduction: Run the whole process of mapping landforms
#
#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 18 January, 2021

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

# set PATH if necessary (e.g. python2 with tensorflow 1.6 on Cryo03)
#export PATH=~/programs/anaconda2/bin:$PATH

# set GPUs want to used (e.g. use GPU 0 & 1)
export CUDA_VISIBLE_DEVICES=0,1
# number of GPUs
gpu_num=2

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
# there is another script ("build_RS_data.py"), but seem have not finished.

${eo_dir}/workflow/split_sub_images.py ${para_file}
${eo_dir}/workflow/training_img_augment.py ${para_file}
${eo_dir}/workflow/split_train_val.py ${para_file}

## convert to TFrecord
${eo_dir}/workflow/build_TFrecord_tf1x.py ${para_file}


duration=$SECONDS
echo "$(date): time cost of preparing training data: ${duration} seconds">>"time_cost.txt"

################################################
## training

${eo_dir}/workflow/deeplab_train.py ${para_file} ${gpu_num}

################################################

#export model
${eo_dir}/workflow/export_graph.py ${para_file}

################################################
## inference
rm -r multi_inf_results || true
${eo_dir}/workflow/parallel_prediction.py ${para_file}


## post processing and copy results, inf_post_note indicate notes for inference and post-processing
inf_post_note=1
${eo_dir}/workflow/postProcess.py ${para_file}  ${inf_post_note}



#################################################
### conduct polygon-based change detection based on the multi-temporal mapping results
#cd_code=~/codes/PycharmProjects/ChangeDet_DL
#${cd_code}/thawSlumpChangeDet/polygons_cd_multi_exe.py ${para_file} ${test_name}
