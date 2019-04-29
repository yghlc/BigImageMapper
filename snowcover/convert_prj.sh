#!/bin/bash

dir=~/Data/Qinghai-Tibet/beiluhe/modis_snow_cover

# get original project info
#gdalsrsinfo -o epsg 2018_05_01.tif

# MODIS Sinusoidal
s_srs=EPSG:54008
# utm 
t_srs=EPSG:32646

res=500

cd ${dir}

for tif in $(ls *.tif | grep -v utm); do

    filename=$(basename "$tif")
    filename_no_ext="${filename%.*}"
    #extension="${filename##*.}"
    out_name=${filename_no_ext}_utm.tif

    gdalwarp -overwrite -r near -s_srs ${s_srs} -t_srs ${t_srs} -tr ${res} ${res} -of GTiff ${tif} ${out_name}

done





