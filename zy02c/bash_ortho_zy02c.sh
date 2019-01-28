#!/bin/bash

# orthorectify zy02c HRC images
# run this script in ~/Data/Qinghai-Tibet/beiluhe/beiluhe_ZY3

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 28 January, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

outdir=ZY02C_HRC_orthorectified

mkdir -p ${outdir}

# function of converting to 8bit using gdal_translate with max and min value.
function ortho_rectify() {
    local folder=${1}


    SECONDS=0

    cd ${folder}
    ~/codes/PycharmProjects/Landuse_DL/zy02c/ortho_rect_zy02c.sh
    cd -

    # mv results
    mv ${folder}/*_ortho*  ${outdir}/.

    #exit
    duration=$SECONDS
    echo "$(date): time cost of orthorectification of ${folder}: ${duration} seconds">>"time_cost.txt"
}

for zy02c_hrc in $(ls -d ZY02C_HRC_E* |grep -v gz ); do
    echo zy02c_hrc

    ortho_rectify zy02c_hrc
done






