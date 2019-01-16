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

time_str=`date +%Y_%m_%d_%H_%M_%S`
echo ${time_str} >> ${log}

#para_file=$1
para_file=para.ini
deeplabRS=~/codes/PycharmProjects/DeeplabforRS
para_py=${deeplabRS}/parameters.py
train_shp_all=$(python2 ${para_py} -p ${para_file} training_polygons )

# function to run training with different data augmentation
function train_kfold_cross_val() {
    #get parameters
    local kvalue=${1}
    local test_num=${2}

    echo kvalue : ${kvalue} >> ${log}
    echo test_num : ${test_num} >> ${log}

    # create k subset of shapefile
    dir=$(dirname "${train_shp_all}")
    filename=$(basename "$train_shp_all")
    filename_no_ext="${filename%.*}"
    dir_sub=${dir}/${kvalue}-fold_cross_val_t${test_num}

    # will save to ${dir_sub}
    # if exist, skip
    if [ ! -d "${dir_sub}" ]; then
        mkdir -p ${dir_sub}
        cd ${dir_sub}
        ${deeplabRS}/get_trianing_polygons.py ${train_shp_all} ${filename} -k ${kvalue}
        cd -
    else
        echo "subset of shapefile already exist, skip creating new" >> ${log}
    fi

    # training on k subset
    for idx in $(seq 1 $kvalue); do
        # remove previous trained model (the setting are the same to exp9)
        rm -r exp9 || true

        echo run training and inference of the ${idx}_th fold >> ${log}

        # modified para.ini
        cp para_template_kfold.ini para.ini
        newline=${dir_sub}/${filename_no_ext}_${kvalue}fold_${idx}.shp
        sed -i -e  s%x_train_polygon_sub%$newline%g para.ini
        # modified exe.sh
        cp exe_template_kfold.sh exe.sh
        newline=${kvalue}fold_${idx}_t${test_num}
        sed -i -e  s%x_test_num%$newline%g exe.sh

        # run
        ./exe.sh

    done
}


#k=5
#train_kfold_cross_val ${k} 1

k=5
train_kfold_cross_val ${k} 2

