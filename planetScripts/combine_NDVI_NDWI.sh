#!/usr/bin/env bash

## Introduction:  Merge NDVI, NDWI, and one RGB band to a three bands image.

name_pre=20180522_035755_3B_AnalyticMS_SR_mosaic

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

# convert NDVI, NDWI to 8 bit
gdal_contrast_stretch -percentile-range 0.01 0.99 ${name_pre}_NDVI.tif ${name_pre}_NDVI_8bit.tif

gdal_contrast_stretch -percentile-range 0.01 0.99 ${name_pre}_NDWI.tif ${name_pre}_NDWI_8bit.tif

# get the first band (it is blue?, or use PCA to get one band from RGB?)
gdal_translate -b 1  ${name_pre}_8bit.tif ${name_pre}_8bit_b1.tif

#combine the first band, NDVI, NDWI
gdal_merge.py -separate -o ${name_pre}_8bit_b1_ndvi_ndwi.tif \
${name_pre}_8bit_b1.tif ${name_pre}_NDVI_8bit.tif ${name_pre}_NDWI_8bit.tif
