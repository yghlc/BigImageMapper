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

./cal_TheilSen_trend.py ${input_tifs}