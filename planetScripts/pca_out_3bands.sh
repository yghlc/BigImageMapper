#!/usr/bin/env bash

## Introduction:  conduct Dimensionality Reduction using otbcli_DimensionalityReduction, help can be found in
# https://www.orfeo-toolbox.org/CookBook/Applications/app_DimensionalityReduction.html

# test
#cd ~/Data/Qinghai-Tibet/beiluhe/beiluhe_planet/beiluhe_basin/201805

#name_pre=20180522_035755_3B_AnalyticMS_SR_mosaic
#name_pre=20180522_035755_3B_AnalyticMS_SR_mosaic_subPCA
name_pre=$1
# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

keep_bands=3
out_min=1.0    # it seems these two options not work, 30 Dec 2018 hlc
out_max=254.0
b_norma=1  # 1 for on, 0 for off

# seems min and max option not work, we cannot set this one as int type
pixel=float #pixel=uint8/uint16/int16/uint32/int32/float/double] (default value is float)

otbcli_DimensionalityReduction -progress 1 -in ${name_pre}.tif -out ${name_pre}_PCA3b.tif ${pixel} -method pca \
-normalize ${b_norma} -rescale.outmin ${out_min} -rescale.outmax ${out_max} \
-nbcomp ${keep_bands} \
-outxml ${name_pre}_save.xml

#  -rescale ${out_min} ${out_max}
# -rescale.outmin ${out_min} -rescale.outmax ${out_max}

#convert to 8 bit, 0-255
gdal_contrast_stretch -percentile-range 0.01 0.99 ${name_pre}_PCA3b.tif ${name_pre}_PCA3b_8bit.tif




