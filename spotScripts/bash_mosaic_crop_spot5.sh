#!/bin/bash

# mosaic of spot5 images if they acquired on the same date, then crop them to beiluhe basin
# run this script in the SPOT 5 folder: e.g.,
# ~/Data/Qinghai-Tibet/beiluhe/beiluhe_spot/spot5_orthorectified

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

#export PATH=~/programs/StereoPipeline-2.6.1-2018-09-06-x86_64-OSX/bin:$PATH
#export PATH=~/programs/StereoPipeline-2.6.1-2018-09-06-x86_64-Linux/bin:$PATH
export PATH=~/programs/StereoPipeline-2.6.1-2019-01-19-x86_64-Linux/bin:$PATH

#number of thread of to use, 8 or 16 on linux, 4 on mac (default is 4)
num_thr=16

out_res=2.5
nodata=0

outdir=../spot5_BLH_extent

mkdir -p ${outdir}

function filename_no_ext(){
    local input=$1
    filename=$(basename input)
    filename_no_ext="${filename%.*}"
    echo $filename_no_ext

}
function mosaic_tow_img(){
    local img1=$1
    local img2=$2

    echo conduct mosaic of $img1 and $img2

    SECONDS=0

    pre_name=$(filename_no_ext $img1)
    out_mos=${pre_name}_mos.tif

    # use gdalwarp to mosaic these two
    # (can choose how to calculate the pixel values in overlap area), better than gdal_merge.py,
    gdalwarp -r average -tr ${out_res} ${out_res} ${img1} ${img2} ${out_mos}.tif

    duration=$SECONDS
    echo "$(date): time cost of mosaic of ${img1} and ${img2}: ${duration} seconds">>"time_cost.txt"

    # return the output path
    echo $out_mos

}

function crop_beiluhe() {
    local image=$1
#    local shp=$2

    SECONDS=0

    pre_name=$(filename_no_ext $image)
    out_crop=${pre_name}_basinExt.tif

    gdalwarp -cutline ~/Data/Qinghai-Tibet/beiluhe/beiluhe_reiver_basin.kml \
        -crop_to_cutline -tr ${out_res} ${out_res} -of GTiff ${image} ${out_crop}

    # move results
    mv ${out_crop} ${outdir}/.

    duration=$SECONDS
    echo "$(date): time cost of crop ${image}: ${duration} seconds">>"time_cost.txt"
}

# Mosaic the images on the same acquisition date
# Crop all the images to Beiluhe extent

#2010-05-06
mos=$(mosaic_tow_img beiluhe_spot5_T_2010-05-06_234-2??.tif)
crop_beiluhe $mos

mos=$(mosaic_tow_img beiluhe_spot5_T-X_2010-05-06_234-2??.tif)
crop_beiluhe $mos

# for other years (no need mosaic)
for tif in $(ls beiluhe_spot5*.tif | grep -v 2010-05-06); do
    crop_beiluhe $tif
done







