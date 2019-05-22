#!/bin/bash

# plot visualization of snow days into one figures

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 22 May, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace


# October from 2003 to 2012
dir=~/Data/Qinghai-Tibet/beiluhe/modis_snow_cover/beiluhe_monthly_snow_days_GooImgExt/Oct2003to2012

output=snow_days_October2003to2012.tif

./plot_vis_snow_day.py $(find ${dir}/*.tif) -o ${output}






