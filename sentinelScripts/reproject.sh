#!/usr/bin/env bash

# convert projection sentinel-2 images downloaded from Google Earth Engine
# authors: Huang Lingcao
# email:huanglingcao@gmail.com
# add time: 29 September, 2019


# run this script in /home/hlc/Data/Qinghai-Tibet/entire_QTP_images/sentinel-2

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

para_py=~/codes/PycharmProjects/DeeplabforRS/parameters.py


para_file=para.ini

# target projection
t_srs=$(python2 ${para_py} -p ${para_file} cartensian_prj)
echo $t_srs

res=10

verdir=sentinel-2_2018_mosaic_v4

out_dir=${verdir}_Albers

mkdir -p 8bit_dir/${out_dir}

for tif in $(ls 8bit_dir/${verdir}/*.tif); do

    # convert projection
    echo "INPUT tif file:" $tif

    s_srs=$(gdalsrsinfo -o wkt $tif )   # could be
    echo "The original EPGS is" ${s_srs}

    filename=$(basename "$tif")
    filename_no_ext="${filename%.*}"
    #extension="${filename##*.}"
    out_name=${filename_no_ext}_Albers.tif

    #
    gdalwarp -overwrite -r bilinear  -s_srs ${s_srs} -t_srs ${t_srs} -tr ${res} ${res} -of GTiff ${tif} ${out_name}


done