#!/usr/bin/env bash

## Introduction:  conduct Dimensionality Reduction using otbcli_DimensionalityReduction, help can be found in
# https://www.orfeo-toolbox.org/CookBook/Applications/app_DimensionalityReduction.html

# test
cd ~/Data/Qinghai-Tibet/beiluhe/beiluhe_planet/beiluhe_basin/201805

#name_pre=20180522_035755_3B_AnalyticMS_SR_mosaic
name_pre=20180522_035755_3B_AnalyticMS_SR_mosaic_subPCA
# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

keep_bands=3
out_min=10.0
out_max=200.0
b_norma=0  # 1 for on, 0 for off
pixel=uint8 #pixel=uint8/uint16/int16/uint32/int32/float/double] (default value is float)

#${pixel}

# -nbcomp ${keep_bands}

otbcli_DimensionalityReduction -progress 1 -in ${name_pre}_8bit.tif -out ${name_pre}_8bit_PCA3b.tif   -method pca \
-normalize ${b_norma}  -rescale.outmin ${out_min} -rescale.outmax ${out_max} \
-outxml ${name_pre}_save.xml

#  -rescale ${out_min} ${out_max}
# -rescale.outmin ${out_min} -rescale.outmax ${out_max}


