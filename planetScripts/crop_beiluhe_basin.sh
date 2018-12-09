#!/usr/bin/env bash

name_pre=20180522_035755_3B_AnalyticMS_SR_mosaic

gdalwarp -q -cutline ~/Data/Qinghai-Tibet/beiluhe/beiluhe_reiver_basin.kml \
-crop_to_cutline -tr 3.0 3.0 -of GTiff ${name_pre}_8bit_rgb.tif ${name_pre}_8bit_rgb_basinExt.tif