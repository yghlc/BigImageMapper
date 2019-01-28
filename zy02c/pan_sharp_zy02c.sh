#!/bin/bash

#
# run this script in the ZY3 image folder contaning NAD images: e.g.,
# ~/Data/Qinghai-Tibet/beiluhe/beiluhe_ZY02C/ZY02C_PMS_E92.7_N34.8_20121108_L1C0000807138

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

#export PATH=~/programs/StereoPipeline-2.6.1-2018-09-06-x86_64-OSX/bin:$PATH
#export PATH=~/programs/StereoPipeline-2.6.1-2018-09-06-x86_64-Linux/bin:$PATH
export PATH=~/programs/StereoPipeline-2.6.1-2019-01-19-x86_64-Linux/bin:$PATH

#number of thread of to use, 8 or 16 on linux, 4 on mac (default is 4)
num_thr=16

#dem=~/Data/Qinghai-Tibet/beiluhe/DEM/srtm_30/beiluhe_strm30.tif
dem=~/Data/Qinghai-Tibet/beiluhe/DEM/srtm_30/beiluhe_strm30_prj_utm.tif

# spatial resolution of output, pan and mss images
out_res=5.0
pan_res=5.0
mss_res=10.0

nodata=0
# one solution: use NAD as the left image, FWD or BWD as the right image?
gray=$(ls *-PAN.tiff)
color=$(ls *-MUX.tiff)

# for test: xmin ymin xmax ymax, should not use the pixel based coordinate, then PAN and MSS have different extent
#test_roi="--t_pixelwin 5000 5000 6000 6000"
test_roi=

str=$(basename $PWD)
IFS='_ ' read -r -a array <<< "$str"
output=${array[0]}_${array[1]}_${array[4]}_${array[5]}

## ortho  first
mapproject -t rpc --nodata-value ${nodata} --tr ${pan_res} ${dem} ${gray} gray_mapped.tif \
        ${test_roi} --threads ${num_thr} --ot Byte --tif-compress None

# Input images must be single channel
#mapproject -t rpc --nodata-value ${nodata} --tr ${mss_res} ${dem} ${color} color_mapped.tif \
#        ${test_roi} --threads ${num_thr} --ot Byte --tif-compress None

# mapproject the MSS file band by band, then merge them.
color_fname=$(basename $color)
color_fname_no_ext="${color_fname%.*}"
for band in $(seq 1 3); do

    echo extract band: $band of $color
    pre_name=${color_fname_no_ext}_B${band}
    gdal_translate -b $band $color ${pre_name}.tif
    cp ${color_fname_no_ext}.rpb ${pre_name}.rpb

    mapproject -t rpc --nodata-value ${nodata} --tr ${mss_res} ${dem} ${pre_name}.tif color_mapped_B${band}.tif \
            ${test_roi} --threads ${num_thr} --ot Byte --tif-compress None

#    exit
done

gdal_merge.py -separate -a_nodata ${nodata} -o color_mapped.tif color_mapped_B?.tif
cp ${color_fname_no_ext}.rpb color_mapped.RPB


## using the pansharpening in OTB
~/codes/PycharmProjects/Landuse_DL/zy3Scripts/crop2theSameExtent.py color_mapped.tif gray_mapped.tif
# method:rcs/lmvm/bayes , rcs has Segmentation fault (core dumped)
otbcli_Pansharpening -progress 1 -method lmvm -inp gray_mapped.tif -inxs color_mapped_crop.tif -out ${output}_otb.tif uint8

#set nondata (not necessary)
#gdal_edit.py -a_nodata ${nodata} ${output}_otb.tif


# the third band is red, second is green, and first is blue
gdal_translate -b 3 -b 2 -b 1  ${output}_otb.tif ${output}_otb_rgb.tif

# get a overview for visualization
otbcli_Quicklook -progress 1 -sr 32 -in ${output}_otb_rgb.tif -out ${output}_otb_rgb_overview.tif uint8