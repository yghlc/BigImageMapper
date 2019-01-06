#!/usr/bin/env bash

# run tests on using SVM classification
# copy this script to the working folder
# e.g., working folder: ~/Data/Qinghai-Tibet/beiluhe/beiluhe_planet/autoMapping/BLH_basin_deeplabV3+_1
# then run

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

### using python2 ###
#MAKE SURE the /usr/bin/python, which is python2 on Cryo06
export PATH=/usr/bin:$PATH
# python2 on Cryo03, tensorflow 1.6
export PATH=/home/hlc/programs/anaconda2/bin:$PATH

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


##########pre-processing for SVM##########
input_image=$(python2 ${para_py} -p ${para_file} input_image_path )
#rm "scaler_saved.pkl" || true
#${eo_dir}/planetScripts/planet_svm_classify.py ${input_image}  -p
#
#
###########svm training##########
#
## remove previous subImages
#${eo_dir}/thawslumpScripts/remove_previous_data.sh ${para_file}
#
## get a ground truth raster if it did not exists or the corresponding shape file gets update
#${eo_dir}/thawslumpScripts/get_ground_truth_raster.sh ${para_file}
#
## get subImages (using four bands) and subLabels for training, extract sub_images based on the training polygons
## make sure ground truth raster already exist
#${eo_dir}/thawslumpScripts/get_sub_images.sh ${para_file}

rm "sk_svm_trained.pkl" || true
${eo_dir}/planetScripts/planet_svm_classify.py -t

###########pclassification##########
${eo_dir}/planetScripts/planet_svm_classify.py ${input_image}

# accuracies assessment
${eo_dir}/thawslumpScripts/accuracies_assess.sh ${para_file}


