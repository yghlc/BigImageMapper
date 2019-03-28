#!/bin/bash

# compare validate the multispectral indices (MSI) from Google Earth Engine
# by comparing to the our calculation

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 24 March, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

gee_msi_dir=~/Data/Qinghai-Tibet/beiluhe/beiluhe_landsat/landsat_multispectral_indices
msi_dir=./

########################################################################
## for landsat 8
## compare ndvi
#./compare_to_gee.py landsat8_ndvi.tif ${gee_msi_dir}/beiluhe_LC08_NDVI_2013to2013.tif
#
## compare ndwi
#./compare_to_gee.py landsat8_ndwi.tif ${gee_msi_dir}/beiluhe_LC08_NDWI_2013to2013.tif

# compare ndmi
#./compare_to_gee.py landsat8_ndmi.tif ${gee_msi_dir}/beiluhe_LC08_NDMI_2013to2013.tif

# compare brightness
#./compare_to_gee.py landsat8_brightness.tif ${gee_msi_dir}/beiluhe_LC08_brightness_2013to2013.tif

#greenness
#./compare_to_gee.py landsat8_greenness.tif ${gee_msi_dir}/beiluhe_LC08_greenness_2013to2013.tif

#wetness
#./compare_to_gee.py landsat8_wetness.tif ${gee_msi_dir}/beiluhe_LC08_wetness_2013to2013.tif

########################################################################
## for landsat 7
sat=landsat7
./compare_to_gee.py ${sat}_ndvi.tif ${gee_msi_dir}/beiluhe_LE07_NDVI_1990to2018.tif
./compare_to_gee.py ${sat}_ndwi.tif ${gee_msi_dir}/beiluhe_LE07_NDWI_1990to2018.tif
./compare_to_gee.py ${sat}_ndmi.tif ${gee_msi_dir}/beiluhe_LE07_NDMI_1990to2018.tif
./compare_to_gee.py ${sat}_brightness.tif ${gee_msi_dir}/beiluhe_LE07_brightness_1990to2018.tif
./compare_to_gee.py ${sat}_greenness.tif ${gee_msi_dir}/beiluhe_LE07_greenness_1990to2018.tif
./compare_to_gee.py ${sat}_wetness.tif ${gee_msi_dir}/beiluhe_LE07_wetness_1990to2018.tif


########################################################################
## for landsat 5
sat=landsat5
#./compare_to_gee.py ${sat}_ndvi.tif ${gee_msi_dir}/beiluhe_LT05_NDVI_2001to2018.tif
#./compare_to_gee.py ${sat}_ndwi.tif ${gee_msi_dir}/beiluhe_LT05_NDWI_2001to2018.tif
#./compare_to_gee.py ${sat}_ndmi.tif ${gee_msi_dir}/beiluhe_LT05_NDMI_2001to2018.tif
#./compare_to_gee.py ${sat}_brightness.tif ${gee_msi_dir}/beiluhe_LT05_brightness_2001to2018.tif
#./compare_to_gee.py ${sat}_greenness.tif ${gee_msi_dir}/beiluhe_LT05_greenness_2001to2018.tif
#./compare_to_gee.py ${sat}_wetness.tif ${gee_msi_dir}/beiluhe_LT05_wetness_2001to2018.tif