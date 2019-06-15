#!/bin/bash

# test calculate Theil-Sen Trend
#

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 12 June, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

gee_msi_dir=~/Data/Qinghai-Tibet/beiluhe/beiluhe_landsat/landsat_multispectral_indices
output_dir=./

input_tifs=$(ls ${gee_msi_dir}/*.tif)


# calculate a trend of one index
function index_trend() {

   index_name=$1

    mkdir -p ${index_name}_trend_patches
    rm ${index_name}_trend_patches/* || true

    ./cal_TheilSen_trend.py ${input_tifs}  --name_index=${index_name}

    gdal_merge.py -o beiluhe_${index_name}_trend.tif  ${index_name}_trend_patches/*.tif

}


index_trend brightness

index_trend greenness

index_trend wetness

index_trend NDVI

index_trend NDWI

index_trend NDMI

