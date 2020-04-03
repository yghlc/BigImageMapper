#!/bin/bash

# co-registration of images
# run this script in ~/Data/Qinghai-Tibet/beiluhe/beiluhe_planet/beiluhe_basin/co_register
# since we utilize "ImageMatchsiftGPU" to automatically find tie-points, which only available on Cryo06

# note that: ImageMatchsiftGPU must be run on Cryo06 through a desktop environment,
# it cannot run through tmate or others

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 3 June, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

# on Cryo06, libs in /home/hlc/programs/anaconda3/lib leads to the crash of ImageMatchsiftGPU, gdal failed to reproject
# so set LD_LIBRARY_PATH manually, 15 Dec 2019
export LD_LIBRARY_PATH=/home/hlc/programs/lib:/home/hlc/programs/cuda-9.0/lib64:/usr/lib/x86_64-linux-gnu/mesa:/usr/lib/x86_64-linux-gnu/dri:/usr/lib/x86_64-linux-gnu/gallium-pipe:/home/hlc/bin/GMT_4.5.14/lib


#ref_img=~/Data/Qinghai-Tibet/beiluhe/beiluhe_planet/beiluhe_basin/201805/20180522_035755_3B_AnalyticMS_SR_mosaic_8bit_rgb_basinExt.tif

# Beiluhe test area
ref_img=~/Data/Qinghai-Tibet/beiluhe/beiluhe_planet/beiluhe_test_area/201805/20180523_3B_AnalyticMS_SR_mosaic_8bit_rgb_sharpen.tif

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

# for the image inside the beiluhe extent
#for img in $(ls ../201?07/*rgb_basinExt.tif |grep -v new ); do

# for the image have been sharpened, we need this to delineate boundaries
for img in $(ls ../201?07/*8bit_rgb_sharpen.tif |grep -v new ); do
    echo $img

    co_register $img
done














