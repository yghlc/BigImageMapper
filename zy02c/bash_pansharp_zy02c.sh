#!/bin/bash

# create dsm from each zy3 scene
# run this script in ~/Data/Qinghai-Tibet/beiluhe/beiluhe_ZY3

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 25 January, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

outdir=ZY02C_PMS_pansharp

mkdir -p ${outdir}

# function of converting to 8bit using gdal_translate with max and min value.
function pan_sharp() {
    local folder=${1}
#    local res=${2}

    cd ${folder}

    SECONDS=0

    ~/codes/PycharmProjects/Landuse_DL/zy02c/pan_sharp_zy02c.sh

    cd -

    # mv results
    mv ${folder}/*_otb*  ${outdir}/.

    #exit
    duration=$SECONDS
    echo "$(date): time cost of extracting dsm of ${folder}: ${duration} seconds">>"time_cost.txt"
}

for zy02c_pms in $(ls -d ZY02C_PMS* |grep -v gz ); do
    echo $zy02c_pms

    pan_sharp $zy02c_pms
done






