#!/bin/bash

# orthorectify using mapproject (aps)
# run this script in the ZY02C image folder contaning NAD images: e.g.,
# ~/Data/Qinghai-Tibet/beiluhe/beiluhe_ZY02C/ZY02C_HRC_E92.7_N34.8_20121108_L1C0000817201

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

#export PATH=~/programs/StereoPipeline-2.6.1-2018-09-06-x86_64-OSX/bin:$PATH
#export PATH=~/programs/StereoPipeline-2.6.1-2018-09-06-x86_64-Linux/bin:$PATH
export PATH=~/programs/StereoPipeline-2.6.1-2019-01-19-x86_64-Linux/bin:$PATH

#number of thread of to use, 8 or 16 on linux, 4 on mac (default is 4)
num_thr=16

#dem=~/Data/Qinghai-Tibet/beiluhe/DEM/srtm_30/beiluhe_strm30.tif
dem=~/Data/Qinghai-Tibet/beiluhe/DEM/srtm_30/beiluhe_strm30_prj_utm.tif

out_res=2.36
nodata=0

str=$(basename $PWD)
IFS='_ ' read -r -a array <<< "$str"
out_pan=${array[0]}_${array[1]}_${array[4]}_${array[5]}_ortho

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

# co-registration them before mosaic
cp ../para.ini .
ref_img=$(ls *-HR1_prj.tif)
new_img=$(ls *-HR2_prj.tif)
rm *_new_warp.tif || true
rm *_new.tif || true
~/codes/PycharmProjects/Landuse_DL/spotScripts/co_register.py ${ref_img} ${new_img} -p para.ini

# use gdalwarp to mosaic these two (can choose how to calculate the pixel values in overlap area), better than gdal_merge.py,
#gdalwarp -r average -tr ${out_res} ${out_res} *_prj.tif ${out_pan}.tif
gdalwarp -r average -tr ${out_res} ${out_res} ${ref_img} *_new_warp.tif ${out_pan}.tif





