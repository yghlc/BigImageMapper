#!/usr/bin/env bash

# run tests on using SVM classification
# copy this script to the working folder
# e.g., working folder: ~/Data/Qinghai-Tibet/beiluhe/beiluhe_planet/autoMapping/BLH_basin_deeplabV3+_1
# then run

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

# output log information to time_cost.txt
#log=test_img_augment_log.txt
log=time_cost.txt
#rm ${log} || true   # or true: don't exit with error and can continue run

time_str=`date +%Y_%m_%d_%H_%M_%S`
echo ${time_str} >> ${log}

eo_dir=~/codes/PycharmProjects/Landuse_DL

#para_file=$1
para_file=para.ini
deeplabRS=~/codes/PycharmProjects/DeeplabforRS
para_py=${deeplabRS}/parameters.py
train_shp_all=$(python2 ${para_py} -p ${para_file} training_polygons )

# get subImages (using four bands) and subLabels, extract sub_images based on the training polgyons
# make sure ground truth raster already exist
${eo_dir}/thawslumpScripts/get_sub_images.sh ${para_file}

#pre-processing for SVM


#svm training

#classification


# accuracies assessment


