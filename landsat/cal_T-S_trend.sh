#!/bin/bash

# test calculate Theil-Sen Trend
#

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 12 June, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

code_dir=~/codes/PycharmProjects/Landuse_DL
gee_msi_dir=~/Data/Qinghai-Tibet/beiluhe/beiluhe_landsat/landsat_multispectral_indices
output_dir=./

input_tifs=$(ls ${gee_msi_dir}/*.tif)

SECONDS=0

# calculate a trend of one index
function index_trend() {

   index_name=$1

    mkdir -p ${index_name}_trend_patches
    rm ${index_name}_trend_patches/* || true

    # the test show that annual single value is not good for calculating the trend.
#    ${code_dir}/landsat/cal_TheilSen_trend.py ${input_tifs}  --name_index=${index_name} --annual_based
    ${code_dir}/landsat/cal_TheilSen_trend.py ${input_tifs}  --name_index=${index_name}

    gdal_merge.py -o beiluhe_${index_name}_trend.tif  ${index_name}_trend_patches/*.tif

}


index_trend brightness
#
#index_trend greenness
#
#index_trend wetness
#
#index_trend NDVI
#
#index_trend NDWI
#
#index_trend NDMI


duration=$SECONDS
echo "$(date): time cost of calculating trends: ${duration} seconds">>"time_cost.txt"