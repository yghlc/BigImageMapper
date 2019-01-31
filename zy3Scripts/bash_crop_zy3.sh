#!/bin/bash

# crop zy3 images to beiluhe basin
# run this script in the folder: e.g.,
# ~/Data/Qinghai-Tibet/beiluhe/beiluhe_ZY3

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

#export PATH=~/programs/StereoPipeline-2.6.1-2018-09-06-x86_64-OSX/bin:$PATH
#export PATH=~/programs/StereoPipeline-2.6.1-2018-09-06-x86_64-Linux/bin:$PATH
export PATH=~/programs/StereoPipeline-2.6.1-2019-01-19-x86_64-Linux/bin:$PATH

#number of thread of to use, 8 or 16 on linux, 4 on mac (default is 4)
num_thr=16
nodata=0

#outdir=../spot5_BLH_extent
#outdir=zy02c_blhzoomin2Ext
outdir=zy3_blhzoomin2Ext

mkdir -p ${outdir}

function filename_no_ext(){
    local input=$1
    filename=$(basename $input)
    filename_no_ext="${filename%.*}"
    echo $filename_no_ext
}

function crop_beiluhe() {
    local image=$1
    local out_res=$2
#    local shp=$2
    #extent=~/Data/Qinghai-Tibet/beiluhe/beiluhe_reiver_basin.kml
    extent=~/Data/Qinghai-Tibet/beiluhe/beiluhe_reiver_basin_extent/beiluhe_zoomIn2.kml

    SECONDS=0

    pre_name=$(filename_no_ext $image)
    out_crop=${pre_name}_blhzoomin2Ext.tif

    # ensure "gdalwarp" is built with KML support, or it will complain cannot open the kml files
    ~/programs/anaconda3/bin/gdalwarp -cutline ${extent} \
    -crop_to_cutline -tr ${out_res} ${out_res} -of GTiff ${image} ${out_crop}

    # move results
    mv ${out_crop} ${outdir}/.

    duration=$SECONDS
    echo "$(date): time cost of crop ${image}: ${duration} seconds">>"time_cost.txt"
}

#for tif in $(ls ZY02C_HRC_orthorectified/*_ortho.tif ); do
#    crop_beiluhe $tif 2.36
#done

#for tif in $(ls ZY02C_PMS_pansharp/*_otb.tif ); do
#    crop_beiluhe $tif 5.0
#done






