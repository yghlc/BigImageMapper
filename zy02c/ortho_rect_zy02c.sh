#!/bin/bash

# orthorectify using mapproject (aps)
# run this script in the ZY3 image folder contaning NAD images: e.g.,
# ~/Data/Qinghai-Tibet/beiluhe/beiluhe_ZY3/ZY3_NAD_E92.8_N35.0_20141207_L1A0002929919

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

export PATH=~/programs/StereoPipeline-2.6.1-2018-09-06-x86_64-OSX/bin:$PATH
#export PATH=~/programs/StereoPipeline-2.6.1-2018-09-06-x86_64-Linux/bin:$PATH

#number of thread of to use, 8 or 16 on linux, 4 on mac (default is 4)
num_thr=4

#dem=~/Data/Qinghai-Tibet/beiluhe/DEM/srtm_30/beiluhe_strm30.tif
dem=~/Data/Qinghai-Tibet/beiluhe/DEM/srtm_30/beiluhe_strm30_prj_utm.tif

out_res=2.36
nodata=0

str=$(basename $PWD)
IFS='_ ' read -r -a array <<< "$str"
out_pan=${array[0]}_${array[1]}_${array[4]}_${array[5]}

#test_roi="--t_pixelwin 0 0 500 500"
test_roi=

for tiff in $(ls *HR?.tiff); do

    echo $tiff
    prename="${tiff%.*}"
    output=${prename}_prj.tif

	# ortho  first
    mapproject -t rpc --nodata-value ${nodata} --tr ${out_res} ${dem} $tiff ${output} \
        ${test_roi} --threads ${num_thr} --ot Byte --tif-compress None

#    exit
done

# use gdalwarp to mosaic these two (can choose how to calculate the pixel values in overlap area), better than gdal_merge.py,

gdalwarp -r average -tr ${out_res} ${out_res} *_prj.tif ${out_pan}.tif





