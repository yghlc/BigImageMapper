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
#export CUDA_VISIBLE_DEVICES=0,1
# number of GPUs
gpu_num=2

# the main parameter files
para_file=main_para.ini

# Landuse_DL scripts dir
eo_dir=~/codes/PycharmProjects/Landuse_DL

################################################
SECONDS=0
# remove previous data or results if necessary
${eo_dir}/workflow/remove_previous_data.py ${para_file}

#extract sub_images based on the training polgyons
${eo_dir}/workflow/get_sub_images_multi_regions.py ${para_file}

exit

################################################
## preparing training images.
# there is another script ("build_RS_data.py"), but seem have not finished.

${eo_dir}/thawslumpScripts/split_sub_images.py ${para_file}
${eo_dir}/thawslumpScripts/training_img_augment.sh ${para_file}

## convert to TFrecord
python ${eo_dir}/datasets/build_muti_lidar_data.py

#exit
duration=$SECONDS
echo "$(date): time cost of preparing training: ${duration} seconds">>"time_cost.txt"
SECONDS=0
################################################
## training

${eo_dir}/grss_data_fusion/deeplab_mutiLidar_train.sh ${para_file} ${gpu_num}

duration=$SECONDS
echo "$(date): time cost of training: ${duration} seconds">>"time_cost.txt"
SECONDS=0
################################################

#export model
${eo_dir}/grss_data_fusion/export_graph.sh ${para_file}


################################################
## inference
rm -r multi_inf_results
${eo_dir}/sentinelScripts/parallel_predict_rts.py ${para_file}

## post processing and copy results, including output "time_cost.txt"
test_name=1
${eo_dir}/sentinelScripts/postProc_qtp.sh ${para_file}  ${test_name}
## merge polygons
${eo_dir}/sentinelScripts/merge_shapefiles.sh ${para_file} ${test_name}

################################################
#${eo_dir}/thawslumpScripts/accuracies_assess.sh ${para_file}

################################################
## conduct polygon-based change detection based on the multi-temporal mapping results
cd_code=~/codes/PycharmProjects/ChangeDet_DL
${cd_code}/thawSlumpChangeDet/polygons_cd_multi_exe.py ${para_file} ${test_name}
