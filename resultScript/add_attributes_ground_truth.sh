#!/usr/bin/env bash

# add attributes to ground truth polygons

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 29 August, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

code_dir=~/codes/PycharmProjects/Landuse_DL
# folder contains results
res_dir=~/Data/Qinghai-Tibet/beiluhe/result/result_paper_mapping_RTS_dl_beiluhe_revised2019

deeplabRS=~/codes/PycharmProjects/DeeplabforRS
para_py=~/codes/PycharmProjects/DeeplabforRS/parameters.py

cd ${res_dir}

# input a parameter: the path of para_file (e.g., para.ini)
para_file=para.ini

if [ ! -f $para_file ]; then
   echo "File ${para_file} not exists in current folder: ${PWD}"
   exit 1
fi

input=~/Data/Qinghai-Tibet/beiluhe/thaw_slumps/train_polygons_for_planet_2018_revised_2019/identified_thawslumps_utm.shp
output=${res_dir}/identified_thawslumps_utm_post.shp

# post processing of shapefile
#cp ../${para_file}  ${para_file}
min_area=$(python2 ${para_py} -p ${para_file} minimum_gully_area)
min_p_a_r=$(python2 ${para_py} -p ${para_file} minimum_ratio_perimeter_area)
${deeplabRS}/polygon_post_process.py -p ${para_file} -a ${min_area} -r ${min_p_a_r} ${input} ${output}

cd -


