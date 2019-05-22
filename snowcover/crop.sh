#!/bin/bash

# crop the a smaller extent (google image extent)

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 22 May, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

imgdir=~/Data/Qinghai-Tibet/beiluhe/modis_snow_cover

extent=~/Data/Qinghai-Tibet/beiluhe/beiluhe_google_img/beiluhe_google_img_extent/beiluhe_google_img_extent.shp


# crop a data set
function crop_to_GooImgExt() {
    local file_pre=${1}
    local daterange=${2}

    name_pre=${file_pre}_${daterange}

    # crop snow product
    gdalwarp -cutline ${extent} \
    -crop_to_cutline -tr 500.0 500.0 -of GTiff ${name_pre}.tif ${name_pre}-GooImgExt.tif
    mv ${name_pre}-GooImgExt.tif ${imgdir}/GooImgExt

    # copy csv file
    cp ${name_pre}.csv ${name_pre}-GooImgExt.csv
    mv ${name_pre}-GooImgExt.csv  ${imgdir}/GooImgExt


    # crop NDSI_Snow_Cover_Class
    name_pre=${file_pre}_NDSI_Snow_Cover_Class_${daterange}
    gdalwarp -cutline ${extent} \
    -crop_to_cutline -tr 500.0 500.0 -of GTiff ${name_pre}.tif ${name_pre}-GooImgExt.tif
    mv ${name_pre}-GooImgExt.tif ${imgdir}/GooImgExt
    cp ${name_pre}.csv ${name_pre}-GooImgExt.csv
    mv ${name_pre}-GooImgExt.csv  ${imgdir}/GooImgExt


    # crop Snow_Albedo_Daily_Tile_Class
    name_pre=${file_pre}_Snow_Albedo_Daily_Tile_Class_${daterange}
    gdalwarp -cutline ${extent} \
    -crop_to_cutline -tr 500.0 500.0 -of GTiff ${name_pre}.tif ${name_pre}-GooImgExt.tif
    mv ${name_pre}-GooImgExt.tif ${imgdir}/GooImgExt
    cp ${name_pre}.csv ${name_pre}-GooImgExt.csv
    mv ${name_pre}-GooImgExt.csv  ${imgdir}/GooImgExt

}

name_pre=${imgdir}/beiluhe_MOD10A1
daterange=2000to2013
crop_to_GooImgExt ${name_pre} ${daterange}


name_pre=${imgdir}/beiluhe_MYD10A1
daterange=2000to2013
crop_to_GooImgExt ${name_pre} ${daterange}







