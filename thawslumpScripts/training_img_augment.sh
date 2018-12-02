#!/bin/bash

echo $(basename $0) : "Perform image augmentation"

# Exit immediately if a command exits with a non-zero status.
set -e

para_file=$1
if [ ! -f "$para_file" ]; then
   echo "File ${para_file} not exists in current folder: ${PWD}"
   exit 1
fi


para_py=~/codes/PycharmProjects/DeeplabforRS/parameters.py

eo_dir=~/codes/PycharmProjects/Landuse_DL

#eo_dir=$(python2 ${para_py} -p ${para_file} codes_dir)
augscript=${eo_dir}/grss_data_fusion/image_augment.py

ignore_classes=$(python2 ${para_py} -p ${para_file} data_aug_ignore_classes)

SECONDS=0

# Helper function to update the list
function update_listfile() {
    # use find instead of ls, to avoid error of "Argument list too long"
    for png in $(find . -maxdepth 1 -type f -name '*.png')
    do
        filename=$(basename "$png")
        filename_no_ext="${filename%.*}"
        echo $filename_no_ext >> "trainval.txt"
    done
    mv "trainval.txt" ../.
}

#######################################################
# don't augmentation ignore classes
if [ ! -z "$ignore_classes" ]
    then
    if [ -d "split_images_tmp" ]; then
        rm -r split_images_tmp
    fi
    if [ -d "split_labels_tmp" ]; then
        rm -r split_labels_tmp
    fi
    mkdir split_images_tmp
    mkdir split_labels_tmp
    mv split_images/*_${ignore_classes}_* split_images_tmp/.
    mv split_labels/*_${ignore_classes}_* split_labels_tmp/.
    cd split_images
    update_listfile
    cd ..
    mv trainval.txt list/.
fi
#######################################################

#augment training images
cd split_images
    ~/programs/anaconda3/bin/python3 ${augscript} -p ../${para_file} -d ./ -e .png ../list/trainval.txt -o ./

    update_listfile
cd ..

#augment training lables
cd split_labels
    ~/programs/anaconda3/bin/python3 ${augscript} -p ../${para_file} -d ./ -e .png --is_ground_truth ../list/trainval.txt -o ./

    # have same list, so we don't need to update again
    #update_listfile

    # in the previous version, we replace the 0 pixel as 255, but forget why,
    # but now, I think it is not necessary
    # replace the 0 pixel as 255
#    for png in $(ls *_R*.png); do
#        ${eo_dir}/grss_data_fusion/remove_zero_pixels.py $png temp.png
#        mv temp.png $png
#    done

cd ..

# copy the training data for elevation
mv trainval.txt list/.
cp list/trainval.txt list/val.txt

#######################################################
# move ignore classes back
if [ ! -z "$ignore_classes" ]
    then
    mv split_images_tmp/* split_images/.
    mv split_labels_tmp/* split_labels/.
    cd split_images
    update_listfile
    cd ..
    mv trainval.txt list/.
    cp list/trainval.txt list/val.txt
fi
#######################################################

duration=$SECONDS
echo "$(date): time cost of preparing training images augmentation: ${duration} seconds">>"time_cost.txt"