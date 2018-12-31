#!/usr/bin/env bash

## Introduction:  conduct Dimensionality Reduction using otbcli_DimensionalityReduction, help can be found in
# https://www.orfeo-toolbox.org/CookBook/Applications/app_DimensionalityReduction.html

# test
cd ~/Data/Qinghai-Tibet/beiluhe/beiluhe_planet/beiluhe_basin/201805

name_pre=20180522_035755_3B_AnalyticMS_SR_mosaic

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

keep_bands=3
out_min=0
out_max=255
b_norma=1  # 1 for on, 0 for off
pixel=uint8 #pixel=uint8/uint16/int16/uint32/int32/float/double] (default value is float)

otbcli_DimensionalityReduction -progress 1 -in ${name_pre}_8bit.tif -out ${name_pre}_8bit_PCA3b.tif ${pixel}  -method pca \
-nbcomp ${keep_bands} -rescale.outmin ${out_min} -rescale.outmax ${out_max} -normalize ${b_norma}


