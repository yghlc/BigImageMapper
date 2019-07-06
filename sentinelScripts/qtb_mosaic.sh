#!/bin/bash

# make the mosaic of QTP image on The Tibetan Plateau
# authors: Huang Lingcao
# email:huanglingcao@gmail.com
# add time: 6 July, 2019

# run this script in /home/hlc/Data/Qinghai-Tibet/entire_QTP_images/sentinel-2

# note: gdalsrsinfo require GDAL >= 2.3

dir_s2=/home/hlc/Data/Qinghai-Tibet/entire_QTP_images/sentinel-2

# get original project info
#gdalsrsinfo -o epsg 2018_05_01.tif

#
#s_srs=EPSG:54008
# utm 46N
t_srs=EPSG:32646

res=10

cd ${dir_s2}

tifs=$(ls gee_saved/*.tif)
output=qtb_sentinel2_2018_mosaic_rgb.tif
out_8bit=qtb_sentinel2_2018_mosaic_rgb_8bit.tif

# it takes too long for making a mosaic of tibetan Plateau.
gdal_merge.py -o ${output} ${tifs}

# to 8bit
gdal_contrast_stretch -percentile-range 0.01 0.99 ${output} ${out_8bit}

#for tif in $(ls *beiluhe*.tif | grep -v utm); do
#
#    # convert projection
#
#    s_srs=$(gdalsrsinfo -o epsg $tif )   # could be
#    echo "The original EPGS is" ${s_srs}
#
#    filename=$(basename "$tif")
#    filename_no_ext="${filename%.*}"
#    #extension="${filename##*.}"
#    out_name=${filename_no_ext}_utm.tif
#
#    gdalwarp -overwrite -r near -s_srs ${s_srs} -t_srs ${t_srs} -tr ${res} ${res} -of GTiff ${tif} ${out_name}
#
#
#    # convert to 8bit
#    out_8bit=${filename_no_ext}_utm_8bit.tif
#    gdal_contrast_stretch -percentile-range 0.01 0.99 ${out_name} ${out_8bit}
#
#done





