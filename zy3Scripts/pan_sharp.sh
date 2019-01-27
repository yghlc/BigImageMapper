#!/bin/bash

#
# run this script in the ZY3 image folder contaning NAD images: e.g.,
# ~/Data/Qinghai-Tibet/beiluhe/beiluhe_ZY3/ZY302_TMS_E92.7_N35.0_20171027_L1A0000345912

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
out_res=2.1
pan_res=2.1
mss_res=5.8

nodata=0
# one solution: use NAD as the left image, FWD or BWD as the right image?
gray=$(ls *-NAD.tiff)
color=$(ls *-MUX.tiff)

# for test: xmin ymin xmax ymax, should not use the pixel based coordinate, then PAN and MSS have different extent
#test_roi="--t_pixelwin 5000 5000 6000 6000"
test_roi=

str=$(basename $PWD)
IFS='_ ' read -r -a array <<< "$str"
output=zy3_${array[4]}_${array[5]}_pansharp

# ortho  first
mapproject -t rpc --nodata-value ${nodata} --tr ${pan_res} ${dem} ${gray} gray_mapped.tif \
        ${test_roi} --threads ${num_thr} --ot UInt16 --tif-compress None

# error: Input images must be single channel or RGB!
#mapproject -t rpc --tr ${mss_res} ${dem} ${color} color_mapped.tif \
#        ${test_roi} --threads ${num_thr} --ot UInt16 --tif-compress None

# mapproject the MSS file band by band, then merge them.
color_fname=$(basename $color)
color_fname_no_ext="${color_fname%.*}"
for band in $(seq 1 4); do

    echo extract band: $band of $color
    pre_name=${color_fname_no_ext}_B${band}
    gdal_translate -b $band $color ${pre_name}.tif
    cp ${color_fname_no_ext}.rpb ${pre_name}.rpb

    mapproject -t rpc --nodata-value ${nodata} --tr ${mss_res} ${dem} ${pre_name}.tif color_mapped_B${band}.tif \
            ${test_roi} --threads ${num_thr} --ot UInt16 --tif-compress None

#    exit
done

gdal_merge.py -separate -o color_mapped.tif color_mapped_B?.tif
cp ${color_fname_no_ext}.rpb color_mapped.RPB


#pansharp --nodata-value 0 --threads ${num_thr} --tif-compress None \
# gray_mapped.tif color_mapped.tif ${output}.tif

# or using the pansharpening in OTB
~/codes/PycharmProjects/Landuse_DL/zy3Scripts/crop2theSameExtent.py color_mapped.tif gray_mapped.tif
# method:rcs/lmvm/bayes , rcs has Segmentation fault (core dumped)
otbcli_Pansharpening -progress 1 -method lmvm -inp gray_mapped.tif -inxs color_mapped_crop.tif -out ${output}_otb.tif uint16

# convert the image for display purpose
gdal_contrast_stretch -percentile-range 0.01 0.99 ${output}_otb.tif ${output}_otb_8bit.tif

# the third band is red, second is green, and first is blue
gdal_translate -b 3 -b 2 -b 1  ${output}_otb_8bit.tif ${output}_otb_8bit_rgb.tif

# get a overview for visualization
otbcli_Quicklook -progress 1 -sr 32 -in ${output}_otb_8bit_rgb.tif -out ${output}_otb_8bit_rgb_overview.tif uint8