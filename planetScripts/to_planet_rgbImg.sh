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
    fin_output=${output}_8bit_rgb_sharpen.tif

    # check weather the output already exist
    if [ -f $fin_output ]; then
        echo "Skip, because File ${fin_output} exists in current folder: ${PWD}"
        return 0
    fi

#    # pre-processing images.
#    gdal_contrast_stretch -percentile-range 0.01 0.99 ${image_path} ${output}_8bit.tif

    # use fix min and max to make the color be consistent to sentinel-images
    src_min=0
    src_max=3000
    dst_min=1       # 0 is the nodata, so set as 1
    dst_max=255
    gdal_translate -ot Byte -scale ${src_min} ${src_max} ${dst_min} ${dst_max} ${image_path} ${output}_8bit.tif

    # the third band is red, second is green, and first is blue
    gdal_translate -b 3 -b 2 -b 1  ${output}_8bit.tif ${output}_8bit_rgb.tif


    # sharpen the image
    code_dir=~/codes/PycharmProjects/Landuse_DL
    /usr/bin/python ${code_dir}/planetScripts/prePlanetImage.py ${output}_8bit_rgb.tif ${fin_output}

    # set nodata
    gdal_edit.py -a_nodata 0  ${fin_output}


#    rm ${output}_8bit.tif
#    rm ${output}_8bit_rgb.tif

    exit 1
}

for tif in $(ls ../*/*_SR.tif); do

    echo $tif
    to_rgb $tif

done



