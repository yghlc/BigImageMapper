#!/bin/bash

# plot visualization of snow days into one figures

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 22 May, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace


# October from 2003 to 2012
#dir=~/Data/Qinghai-Tibet/beiluhe/modis_snow_cover/beiluhe_monthly_snow_days_GooImgExt/Oct2003to2012
#output=snow_days_October2003to2012.tif

#
#dir=~/Data/Qinghai-Tibet/beiluhe/modis_snow_cover/beiluhe_monthly_snow_days_GooImgExt/Nov2003to2012
#output=snow_days_November2003to2012.tif


#dir=~/Data/Qinghai-Tibet/beiluhe/modis_snow_cover/beiluhe_monthly_snow_days_GooImgExt/Dec2003to2012
#output=snow_days_December2003to2012.tif

#dir=~/Data/Qinghai-Tibet/beiluhe/modis_snow_cover/beiluhe_monthly_snow_days_GooImgExt/Jan2003to2012
#output=snow_days_January2003to2012.tif


#dir=~/Data/Qinghai-Tibet/beiluhe/modis_snow_cover/beiluhe_monthly_snow_days_GooImgExt/Feb2003to2012
#output=snow_days_Febuary2003to2012.tif

#dir=~/Data/Qinghai-Tibet/beiluhe/modis_snow_cover/beiluhe_monthly_snow_days_GooImgExt/Mar2003to2012
#output=snow_days_March2003to2012.tif

#dir=~/Data/Qinghai-Tibet/beiluhe/modis_snow_cover/beiluhe_monthly_snow_days_GooImgExt/Apr2003to2012
#output=snow_days_April2003to2012.tif

#dir=~/Data/Qinghai-Tibet/beiluhe/modis_snow_cover/beiluhe_monthly_snow_days_GooImgExt/May2003to2012
#output=snow_days_May2003to2012.tif

#dir=~/Data/Qinghai-Tibet/beiluhe/modis_snow_cover/beiluhe_monthly_snow_days_GooImgExt/Jun2003to2012
#output=snow_days_June2003to2012.tif

#dir=~/Data/Qinghai-Tibet/beiluhe/modis_snow_cover/beiluhe_monthly_snow_days_GooImgExt/Jul2003to2012
#output=snow_days_July2003to2012.tif

#dir=~/Data/Qinghai-Tibet/beiluhe/modis_snow_cover/beiluhe_monthly_snow_days_GooImgExt/Aug2003to2012
#output=snow_days_August2003to2012.tif

dir=~/Data/Qinghai-Tibet/beiluhe/modis_snow_cover/beiluhe_monthly_snow_days_GooImgExt/Sep2003to2012
output=snow_days_September2003to2012.tif

./plot_vis_snow_day.py $(find ${dir}/*.tif) -o ${output}






