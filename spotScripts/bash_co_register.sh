#!/bin/bash

# co-registration of images
# run this script in ~/Data/Qinghai-Tibet/beiluhe/beiluhe_spot/georeference_tif
# since we utilize "ImageMatchsiftGPU" to automatically find tie-points, which only available on Cryo06

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 29 January, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace


ref_img='beiluhe_spot5_pan_20061109.tif'
#new_img='../beiluhe_spot5_pan_20090501.tif'
para_file=para.ini


# function of converting to 8bit using gdal_translate with max and min value.
function co_register() {
    local new_img=${1}
#    local res=${2}

    filename=$(basename $new_img)
    filename_no_ext="${filename%.*}"
    mkdir -p $filename_no_ext

    cd $filename_no_ext
    SECONDS=0
    ~/codes/PycharmProjects/Landuse_DL/spotScripts/co_register.py ${ref_img} ${new_img} -p ../${para_file}

    cd -

    # mv results
    mv $filename_no_ext/*_warp.tif  .

    #exit
    duration=$SECONDS
    echo "$(date): time cost of co-registration for ${new_img}: ${duration} seconds">>"time_cost.txt"
}


for img in $(ls -d beiluhe_spot5*.tif |grep -v new | grep -v 20061109); do
    echo $img

    co_register $img
done














