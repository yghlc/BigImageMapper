#!/usr/bin/env bash

# change the image augmentation strategy, then repeat training, see what the result would be? different?
# copy this script to the working folder
# e.g., working folder on Cryo03: ~/Data/Qinghai-Tibet/beiluhe/beiluhe_planet/autoMapping/BLH_basin_deeplabV3+_1
# then run

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

# output log information to time_cost.txt
#log=test_img_augment_log.txt
log=time_cost.txt
#rm ${log} || true   # or true: don't exit with error and can continue run

time_str=`date +%Y_%m_%d_%H_%M_%S`
echo ${time_str} >> ${log}

# function to run training with different data augmentation
function train_img_aug() {
    #get parameters
    local img_aug_str=${1}
    local test_num=${2}

    echo img_aug_str:${img_aug_str} >> ${log}
    echo test_num:${test_num} >> ${log}

    # remove previous trained model (the setting are the same to exp9)
    rm -r exp9 || true

    # modified para.ini
    cp para_template_imgAug.ini para.ini
    sed -i -e  s/x_img_aug_str/${img_aug_str}/g para.ini
    # modified exe.sh
    cp exe_template_imgAug.sh exe.sh
    sed -i -e  s/x_test_num/imgAug${test_num}/g exe.sh

    # run
    ./exe.sh

}

# spaces are not allow in img_aug_str
#flip, blur, crop, scale, rotate
# 1
train_img_aug flip 1

train_img_aug flip,blur 2

train_img_aug flip,blur,crop 3

train_img_aug flip,blur,crop,scale 4

train_img_aug flip,blur,crop,scale,rotate 5

train_img_aug flip,rotate 6

train_img_aug flip,crop,scale 7


