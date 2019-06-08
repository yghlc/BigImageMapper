#!/usr/bin/env bash

name_pre=example

# output resolution is 0.5 degree
out_res=0.5

outline=~/Data/Qinghai-Tibet/Qinghai-Tibet_Plateau/Qinghai-Tibet_Plateau_outline2_gee/QTP_outline_simplified_0.1.shp

rm ${name_pre}_TP.tif || true

gdalwarp -cutline ${outline} -crop_to_cutline -tr ${out_res} ${out_res} -of GTiff ${name_pre}.tif ${name_pre}_TP.tif

