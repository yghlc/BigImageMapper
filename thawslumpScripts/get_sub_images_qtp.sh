#!/usr/bin/env bash

echo $(basename $0) : "extract sub-images and sub-labels for a given shape file (training polygons)"
#introduction: extract sub-images and sub-labels for a given shape file (training polygons)
#               The training data already,
#               if there are multiple files, then modify para.ini (using sed), run this again,
#               then add the sub-images to "subImages" folder
#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 30 September, 2019


# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

para_file=$1
if [ ! -f $para_file ]; then
   echo "File ${para_file} not exists in current folder: ${PWD}"
   exit 1
fi

deeplabRS=~/codes/PycharmProjects/DeeplabforRS
para_py=${deeplabRS}/parameters.py
eo_dir=~/codes/PycharmProjects/Landuse_DL

all_train_shp=$(python2 ${para_py} -p ${para_file} training_polygons )
train_shp=$(python2 ${para_py} -p ${para_file} training_polygons_sub )

input_image_dir=$(python2 ${para_py} -p ${para_file} input_image_dir )

subImage_dir=$(python2 ${para_py} -p ${para_file} input_train_dir)
subLabel_dir=$(python2 ${para_py} -p ${para_file} input_label_dir)

mkdir -p ${subImage_dir}
mkdir -p ${subLabel_dir}

#
dstnodata=$(python2 ${para_py} -p ${para_file} dst_nodata)
buffersize=$(python2 ${para_py} -p ${para_file} buffer_size)
rectangle_ext=$(python2 ${para_py} -p ${para_file} b_use_rectangle)

${eo_dir}/sentinelScripts/get_subImages.py -f ${all_train_shp} -b ${buffersize} -e .tif \
            -o ${PWD} -n ${dstnodata} -r ${train_shp} ${input_image_dir}
