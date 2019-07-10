#!/usr/bin/env bash

# set nodata for images
nodata=0

#for tif in $(ls *.tif); do
#
#    gdal_edit.py -a_nodata ${nodata} ${tif}
#done


tifs=$(ls sentinel-2_2018_mosaic_v2/*.tif)
output=qtb_sentinel2_2018_mosaic_rgb_8bit_300m.tif
#org_res=0.000089831528412
res=0.0026949 # org_res*30

gdal_merge.py -o ${output} -a_nodata 0 -ps ${res} ${res}  ${tifs}