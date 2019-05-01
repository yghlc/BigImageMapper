#!/bin/bash

# calculate the snow cover for the whole area

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 1 May, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

upcodes.sh

imgdir=~/Data/Qinghai-Tibet/beiluhe/modis_snow_cover

./plot_snowcover_timeseries.py ${imgdir}/beiluhe_MOD10A1_2000to2013.tif ${imgdir}/beiluhe_MYD10A1_2000to2013.tif










