#!/usr/bin/env bash

# run tests on k-fold cross validation
# copy this script to the working folder
# e.g., working folder: ~/Data/Qinghai-Tibet/beiluhe/beiluhe_planet/autoMapping/BLH_basin_deeplabV3+_1
# then run

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

# output log information to time_cost.txt
#log=test_img_augment_log.txt
log=time_cost.txt
#rm ${log} || true   # or true: don't exit with error and can continue run

#time_str=`date +%Y_%m_%d_%H_%M_%S`
#echo ${time_str} on $(hostname) >> ${log}

#para_file=$1
para_file=para.ini
deeplabRS=~/codes/PycharmProjects/DeeplabforRS
para_py=${deeplabRS}/parameters.py
train_shp_all=$(python2 ${para_py} -p ${para_file} training_polygons )


# input k and test_number from outsize
k=$1
test_num=$2


# plot Precision-recall curve
figdir=result_backup/${k}fold_test${test_num}
figname=p_r_${k}fold_test${test_num}.jpg
mkdir -p ${figdir}

mv result_backup/*${k}fold_*_t${test_num}* ${figdir}/.
mv ${log} ${figdir}/.

shp_list=$(ls ${figdir}/*post_${k}fold_*_t${test_num}*.shp)
${deeplabRS}/plot_accuracies.py -p ${para_file} ${shp_list} -o ${figdir}/${figname}




