#!/bin/bash

#  crop spot images to the google image extent
# run this script in the SPOT 5 folder: e.g.,
# ~/Data/Qinghai-Tibet/beiluhe/beiluhe_spot/spot5_blhGooImgExt

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

out_res=2.5
nodata=0

function filename_no_ext(){
    local input=$1
    filename=$(basename $input)
    filename_no_ext="${filename%.*}"
    echo $filename_no_ext

}

function crop_beiluhe() {
    local image=$1
#    local shp=$2
    #extent=~/Data/Qinghai-Tibet/beiluhe/beiluhe_reiver_basin.kml
    #extent=~/Data/Qinghai-Tibet/beiluhe/beiluhe_reiver_basin_extent/beiluhe_zoomIn2.kml

    extent=~/Data/Qinghai-Tibet/beiluhe/beiluhe_google_img/beiluhe_google_img_extent/beiluhe_google_img_extent.shp

    SECONDS=0

    pre_name=$(filename_no_ext $image)
    out_crop=${pre_name}_GooImgExt.tif

    # ensure "gdalwarp" is built with KML support, or it will complain cannot open the kml files
    ~/programs/anaconda3/bin/gdalwarp -cutline ${extent} \
    -crop_to_cutline -tr ${out_res} ${out_res} -of GTiff ${image} ${out_crop}

    duration=$SECONDS
    echo "$(date): time cost of crop ${image}: ${duration} seconds">>"time_cost.txt"
}

#for tif in $(ls ../spot5_blhzoomin2Ext/*2011*blhzoomin2Ext.tif ); do
#    crop_beiluhe $tif
#done

# for other years
for tif in $(ls ../spot5_blhzoomin2Ext/*blhzoomin2Ext.tif | grep -v 2011); do
    crop_beiluhe $tif
done







