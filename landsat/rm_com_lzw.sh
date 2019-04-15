#!/bin/bash

# remove the COMPRESSION=LZW in tif file
# by comparing to the our calculation

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 15 April, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

dir=~/Data/Qinghai-Tibet/beiluhe/beiluhe_landsat/landsat_multispectral_indices

cd ${dir}

for tif in $(ls *.tif); do

    gdal_translate -co "COMPRESSION=None" $tif  tmp.tif
    mv tmp.tif $tif

done









