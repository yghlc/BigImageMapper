#!/bin/bash

# Create RGB images
#

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 27 March, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

gee_msi_dir=~/Data/Qinghai-Tibet/beiluhe/beiluhe_landsat/landsat_multispectral_indices
msi_dir=./

########################################################################
## for landsat 8
#B=${gee_msi_dir}/beiluhe_LC08_brightness_2013to2018.tif
#G=${gee_msi_dir}/beiluhe_LC08_greenness_2013to2018.tif
#W=${gee_msi_dir}/beiluhe_LC08_wetness_2013to2018.tif

#./compose_RGB.py ${B} ${G} ${W}

########################################################################
## for landsat 7
#B=${gee_msi_dir}/beiluhe_LE07_brightness_1990to2018.tif
#G=${gee_msi_dir}/beiluhe_LE07_greenness_1990to2018.tif
#W=${gee_msi_dir}/beiluhe_LE07_wetness_1990to2018.tif

#./compose_RGB.py ${B} ${G} ${W}


########################################################################
## for landsat 5
B=${gee_msi_dir}/beiluhe_LT05_brightness_2001to2018.tif
G=${gee_msi_dir}/beiluhe_LT05_greenness_2001to2018.tif
W=${gee_msi_dir}/beiluhe_LT05_wetness_2001to2018.tif
# compare brightness
./compose_RGB.py ${B} ${G} ${W}
