#!/bin/bash

## Introduction:  derive RGB images from Planet Surface Reflectance

# run under folder:
#/home/hlc/Data/Qinghai-Tibet/entire_QTP_images/planet_sr_images/planet_rgb

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 5 October, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

function to_rgb(){

    image_path=$1

    filename=$(basename "$image_path")
    filename_no_ext="${filename%.*}"

    output=${filename_no_ext}

    # pre-processing images.
    gdal_contrast_stretch -percentile-range 0.01 0.99 ${output}.tif ${output}_8bit.tif

    # the third band is red, second is green, and first is blue
    gdal_translate -b 3 -b 2 -b 1  ${output}_8bit.tif ${output}_8bit_rgb.tif


    # sharpen the image
    code_dir=~/codes/PycharmProjects/Landuse_DL
    /usr/bin/python ${code_dir}/planetScripts/prePlanetImage.py ${output}_8bit_rgb.tif ${output}_8bit_rgb_sharpen.tif

    rm ${output}_8bit.tif
    rm ${output}_8bit_rgb.tif
}

for tif in $(ls ../*/*_SR.tif); do

    echo $tif
    to_rgb $tif

done



