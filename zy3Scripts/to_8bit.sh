#!/bin/bash

# convert the NAD orthorectifieced output from envi to 8bit (tif)
# run this script in ~/Data/Qinghai-Tibet/beiluhe/beiluhe_ZY3

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 14 January, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

outdir=zy3_nad_orthorectified_8bit

mkdir -p ${outdir}

# function of converting to 8bit using gdal_translate with max and min value.
function to8bit_tif() {
    local input=${1}

     # get file name without extension
#    DIR=$(dirname "${input}")
    filename=$(basename "$input")
#    extension="${filename##*.}"
    filename_no_ext="${filename%.*}"

    # convert the image for display purpose
    output=${outdir}/${filename_no_ext}_8bit.tif
    gdal_contrast_stretch -percentile-range 0.01 0.99 ${input} ${output}
}

for zy3_nad in $(ls ZY3_NAD*/*.dat); do
    echo $zy3_nad

    to8bit_tif $zy3_nad
done

for zy3_nad in $(ls ZY302_TMS*/*.dat); do
    echo $zy3_nad

    to8bit_tif $zy3_nad
done






