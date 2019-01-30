#!/bin/bash

# co-registration of images
# run this script in ~/Data/Qinghai-Tibet/beiluhe/beiluhe_spot
# since we utilize "ImageMatchsiftGPU" to automatically find tie-points, which only available on Cryo06

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 29 January, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace


ref_img='../beiluhe_spot5_pan_20061109.tif'
new_img='../beiluhe_spot5_pan_20090501.tif'
para_file=para.ini

~/codes/PycharmProjects/Landuse_DL/spotScripts/co_register.py ${ref_img} ${new_img} -p ${para_file}











