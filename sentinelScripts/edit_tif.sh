#!/usr/bin/env bash

# set nodata for images
nodata=0

#for tif in $(ls *.tif); do
#
#    gdal_edit.py -a_nodata ${nodata} ${tif}
#done


#######################################
# create a mosaic with a coarse resolution (300 meters)
tifs=$(ls sentinel-2_2018_mosaic_v2/*.tif)
output=qtb_sentinel2_2018_mosaic_rgb_8bit_300m.tif
#org_res=0.000089831528412
res=0.0026949 # org_res*30
# get a mosaic of res = 300 m
gdal_merge.py -o ${output} -a_nodata 0 -ps ${res} ${res}  ${tifs}

###################################################
# create overview for each files.
# -ro: create external overview
for tif in $(ls sentinel-2_2018_mosaic_v2/qtb_*_8bit.tif); do
    gdaladdo -ro $tif 4 8 16 32

done