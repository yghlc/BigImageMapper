#!/bin/bash

# pre-processing images.

# images were acquired on 20, 24, 25 July 2018. Most of them is on 25 July
output=20180725_3B_AnalyticMS_SR_mosaic
# mosaic the surface reflectance
gdal_merge.py -o ${output}.tif *_3B_AnalyticMS_SR.tif

# convert the image for display purpose
gdal_contrast_stretch -percentile-range 0.01 0.99 ${output}.tif ${output}_8bit.tif 

# the third band is red, second is green, and first is blue
gdal_translate -b 3 -b 2 -b 1  ${output}_8bit.tif ${output}_8bit_rgb.tif

# display on Google Earth
#gdal_translate -of KMLSUPEROVERLAY ${output}_8bit_rgb.tif  ${output}_8bit_rgb.kmz


# sharpen the image
code_dir=~/codes/PycharmProjects/Landuse_DL
/usr/bin/python ${code_dir}/datasets/prePlanetImage.py ${output}_8bit_rgb.tif ${output}_8bit_rgb_sharpen.tif

# to other format
gdal_translate -of KMLSUPEROVERLAY ${output}_8bit_rgb_sharpen.tif  ${output}_8bit_rgb.kmz
gdal_translate -of JPEG ${output}_8bit_rgb_sharpen.tif ${output}_8bit_rgb_sharpen.jpg
#gdal_translate -of PNG ${output}_8bit_rgb_sharpen.tif ${output}_8bit_rgb_sharpen.png

# calculate NDVI
/usr/bin/python ${code_dir}/datasets/planet_NDVI.py ${output}.tif ${output}_NDVI.tif

# calculate NDWI
/usr/bin/python ${code_dir}/datasets/planet_NDWI.py ${output}.tif ${output}_NDWI.tif
