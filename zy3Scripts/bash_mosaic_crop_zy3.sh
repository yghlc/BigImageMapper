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

#outdir=../spot5_BLH_extent
#outdir=zy02c_blhzoomin2Ext
outdir=zy3_blhzoomin2Ext

mkdir -p ${outdir}

function filename_no_ext(){
    local input=$1
#    filename=$(basename $input)
    path_no_ext="${input%.*}"
    echo $path_no_ext
}

function mosaic_two_img(){
    local img1=$1
    local img2=$2
    local out_res=$3
    # nodata
    local nan=$4

    echo "conduct mosaic of" $img1 and $img2 >> "time_cost.txt"

    SECONDS=0

    pre_name=$(filename_no_ext $img1)
    out_mos=${pre_name}_mos.tif

    # use gdalwarp to mosaic these two
    # (can choose how to calculate the pixel values in overlap area), better than gdal_merge.py,
    gdalwarp -srcnodata ${nan} -dstnodata ${nan} -r average -tr ${out_res} ${out_res} \
        ${img1} ${img2} ${out_mos}

    duration=$SECONDS
    echo "$(date): time cost of mosaic of ${img1} and ${img2}: ${duration} seconds">>"time_cost.txt"

    # return the output path
    echo $out_mos

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

# Mosaic the images on the same acquisition date
# Crop all the images to Beiluhe extent

#20141207
res=2.1
nodata=0
#mos=$(mosaic_two_img zy3_nad_ortho/ZY3_20141207*_8bit.tif ${res} ${nodata} )
#crop_beiluhe $mos

# for other years (no need mosaic)
#for tif in $(ls zy3_nad_ortho/ZY3*_8bit.tif ); do
#    crop_beiluhe $tif ${res}
#done


# dem
res=3.0
nodata=-9999
# for other years (no need mosaic)
for tif in $(ls zy3_dsm_files/zy3_*.tif ); do
    crop_beiluhe $tif ${res}
done


# pansharp
res=2.1
nodata=0
# for other years (no need mosaic)
for tif in $(ls ZY302_TMS_pansharp/ZY3*_otb_8bit_rgb.tif ); do
    crop_beiluhe $tif ${res}
done







