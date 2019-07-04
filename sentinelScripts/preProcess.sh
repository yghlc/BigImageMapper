#!/bin/bash

# convert projection, bit depth of images downloaded from Google Earth Engine
# authors: Huang Lingcao
# email:huanglingcao@gmail.com
# add time: 4 July, 2019

# note: gdalsrsinfo require GDAL >= 2.3

dir_s2=~/Data/Qinghai-Tibet/beiluhe/beiluhe_sentinel-2
dir_s1=~/Data/Qinghai-Tibet/beiluhe/beiluhe_sentinel-1

# get original project info
#gdalsrsinfo -o epsg 2018_05_01.tif

#
#s_srs=EPSG:54008
# utm 46N
t_srs=EPSG:32646

res=10

#cd ${dir_s2}
cd ${dir_s1}


for tif in $(ls *beiluhe*.tif | grep -v utm); do

    # convert projection

    s_srs=$(gdalsrsinfo -o epsg $tif )   # could be
    echo "The original EPGS is" ${s_srs}

    filename=$(basename "$tif")
    filename_no_ext="${filename%.*}"
    #extension="${filename##*.}"
    out_name=${filename_no_ext}_utm.tif

    gdalwarp -overwrite -r near -s_srs ${s_srs} -t_srs ${t_srs} -tr ${res} ${res} -of GTiff ${tif} ${out_name}


    # convert to 8bit
    out_8bit=${filename_no_ext}_utm_8bit.tif
    gdal_contrast_stretch -percentile-range 0.01 0.99 ${out_name} ${out_8bit}

done





