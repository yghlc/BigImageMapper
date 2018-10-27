#!/bin/bash


#introduction: Run the whole process of mapping thaw slumps base on DeeplabV3+
#
#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 26 October, 2018


#MAKE SURE the /usr/bin/python, which is python2 on Cryo06
export PATH=/usr/bin:$PATH

# Exit immediately if a command exits with a non-zero status.
set -e

eo_dir=/home/hlc/codes/PycharmProjects/Landuse_DL
cd ${eo_dir}
git pull
cd -

## modify according to test requirement or environment
#set GPU on Cryo06
export CUDA_VISIBLE_DEVICES=1
para_file=para.ini

################################################
SECONDS=0
## get a ground truth raster if it did not exists or the corresponding shape file gets update
#TODO: insert a script


################################################
## preparing training images.
# there is another script ("build_RS_data.py"), but seem have not finished.  26 Oct 2018 hlc

${eo_dir}/grss_data_fusion/preparing_traindata.sh ${para_file}
${eo_dir}/grss_data_fusion/training_img_augment.sh

## convert to TFrecord
python ${eo_dir}/datasets/build_muti_lidar_data.py

#exit
duration=$SECONDS
echo "$(date): time cost of preparing training: ${duration} seconds">>"time_cost.txt"
SECONDS=0
################################################
## training

${eo_dir}/grss_data_fusion/deeplab_mutiLidar_train.sh

duration=$SECONDS
echo "$(date): time cost of training: ${duration} seconds">>"time_cost.txt"
SECONDS=0
################################################

#export model
${eo_dir}/grss_data_fusion/export_graph.sh ${para_file}


################################################
## inference and post processing, including output "time_cost.txt"
${eo_dir}/thawslumpScripts/inf_postProc.sh ${para_file}


