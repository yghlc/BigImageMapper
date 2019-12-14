#!/bin/bash

# co-registration of images
# run this script in ~/Data/Qinghai-Tibet/beiluhe/beiluhe_planet/beiluhe_basin/co_register
# since we utilize "ImageMatchsiftGPU" to automatically find tie-points, which only available on Cryo06

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 3 June, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace


ref_img=~/Data/Qinghai-Tibet/beiluhe/beiluhe_planet/beiluhe_basin/201805/20180522_035755_3B_AnalyticMS_SR_mosaic_8bit_rgb_basinExt.tif

#ref_img=~/Data/Qinghai-Tibet/beiluhe/beiluhe_google_img/beiluhe_google_img_zoomIn2_2010_utm.tif

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
    ~/codes/PycharmProjects/Landuse_DL/spotScripts/co_register.py ${ref_img} ../${new_img} -p ../${para_file}

    cd ..


    #exit
    duration=$SECONDS
    echo "$(date): time cost of co-registration for ${new_img}: ${duration} seconds">>"time_cost.txt"
}


for img in $(ls ../201?07/*rgb_basinExt.tif |grep -v new ); do
    echo $img

    co_register $img
done














