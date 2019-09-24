#!/bin/bash

# conver to 8bit, file by file
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

verdir=sentinel-2_2018_mosaic_v4

mkdir -p 8bit_dir/${verdir}

for tif in $(ls gee_saved/${verdir}/*.tif); do

    # convert projection
    filename=$(basename "$tif")
    filename_no_ext="${filename%.*}"
    #extension="${filename##*.}"
#    out_name=${filename_no_ext}_8bit.tif


    # convert to 8bit
    out_8bit=8bit_dir/${verdir}/${filename_no_ext}_8bit.tif
#    gdal_contrast_stretch -percentile-range 0.01 0.99 ${out_name} ${out_8bit}

    # gdal_translate to make max, and min consistant over files
    gdal_translate -scale 0 3000 0 255 -ot Byte $tif ${out_8bit}

    # for test
#    exit
done





