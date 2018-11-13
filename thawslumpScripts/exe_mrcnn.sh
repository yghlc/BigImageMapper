#!/bin/bash


#introduction: Run the whole process of mapping thaw slumps base on Mask_RCNN
#
#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 11 November, 2018


#MAKE SURE the /usr/bin/python, which is python2 on Cryo06
export PATH=/usr/bin:$PATH
# python2 on Cryo03, tensorflow 1.6
export PATH=/home/hlc/programs/anaconda2/bin:$PATH

# Exit immediately if a command exits with a non-zero status.
set -e

eo_dir=~/codes/PycharmProjects/Landuse_DL
cd ${eo_dir}
git pull
cd -

cd ~/codes/PycharmProjects/object_detection/yghlc_Mask_RCNN
git pull
cd -


## modify according to test requirement or environment
#set GPU on Cryo06
export CUDA_VISIBLE_DEVICES=1
#set GPU on Cryo03
export CUDA_VISIBLE_DEVICES=0,1
gpu_num=2
para_file=para.ini

################################################
SECONDS=0
## remove previous data or results if necessary
#${eo_dir}/thawslumpScripts/remove_previous_data.sh ${para_file}
#
## get a ground truth raster if it did not exists or the corresponding shape file gets update
#${eo_dir}/thawslumpScripts/get_ground_truth_raster.sh ${para_file}
#
##extract sub_images based on the training polgyons
#${eo_dir}/thawslumpScripts/get_sub_images.sh ${para_file}

################################################
## preparing training images.
# there is another script ("build_RS_data.py"), but seem have not finished.  26 Oct 2018 hlc

#${eo_dir}/thawslumpScripts/split_sub_images.sh ${para_file}
##${eo_dir}/thawslumpScripts/training_img_augment.sh

#TODO: separate the images to training and validation portion.


#exit
duration=$SECONDS
echo "$(date): time cost of preparing training: ${duration} seconds">>"time_cost.txt"
SECONDS=0
################################################
## training

python3 ${eo_dir}/thawslumpScripts/thawS_rs_maskrcnn.py train \
    --para_file=${para_file} \
    --model='coco'

duration=$SECONDS
echo "$(date): time cost of training: ${duration} seconds">>"time_cost.txt"
SECONDS=0
################################################


################################################
## inference and post processing, including output "time_cost.txt"
#${eo_dir}/thawslumpScripts/thawS_rs_maskrcnn.py ????


################################################
## backup results
#${eo_dir}/thawslumpScripts/backup_results.sh ${para_file} 1