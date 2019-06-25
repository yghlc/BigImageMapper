#!/bin/bash

# compose one trend file, each band contain one Theil-Sen Trend
#

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 25 June, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

trend_dir=~/Data/Qinghai-Tibet/beiluhe/beiluhe_landsat/landsat_trend
output_dir=./

input_tifs=$(ls ${trend_dir}/*.tif)

brightness_trend=${trend_dir}/beiluhe_brightness_trend.tif

greenness_trend=${trend_dir}/beiluhe_greenness_trend.tif

wetness_trend=${trend_dir}/beiluhe_wetness_trend.tif

NDVI_trend=${trend_dir}/beiluhe_NDVI_trend.tif

NDWI_trend=${trend_dir}/beiluhe_NDWI_trend.tif

NDMI_trend=${trend_dir}/beiluhe_NDMI_trend.tif

# extract one band
function extract_one_band() {

    band_index=$1
    src_file=$2
    tmp_dir=$3

    filename=$(basename "$src_file")
    filename_no_ext="${filename%.*}"

    gdal_translate -of VRT -b ${band_index} ${src_file} ${tmp_dir}/${filename_no_ext}_b${band_index}.tif

}

# compose a six band file contianing the first trend slope
band_index=1
tmp_dir=band_${band_index}
mkdir -p ${tmp_dir}
rm ${tmp_dir}/* || true

extract_one_band ${band_index} ${brightness_trend} ${tmp_dir}
extract_one_band ${band_index} ${greenness_trend} ${tmp_dir}
extract_one_band ${band_index} ${wetness_trend} ${tmp_dir}
extract_one_band ${band_index} ${NDVI_trend} ${tmp_dir}
extract_one_band ${band_index} ${NDWI_trend} ${tmp_dir}
extract_one_band ${band_index} ${NDMI_trend} ${tmp_dir}

gdal_merge.py -separate -o beiluhe_6msi_trend_slope_${band_index}.tif  ${tmp_dir}/*brightness*.tif \
${tmp_dir}/*greenness*.tif ${tmp_dir}/*wetness*.tif ${tmp_dir}/*NDVI*.tif ${tmp_dir}/*NDWI*.tif \
${tmp_dir}/*NDMI*.tif



