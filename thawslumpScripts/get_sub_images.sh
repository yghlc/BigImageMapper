#!/usr/bin/env bash

echo $(basename $0) : "extract sub-images and sub-labels for a given shape file (training polygons)"
#introduction: extract sub-images and sub-labels for a given shape file (training polygons)
#               The training data already,
#               if there are multiple files, then modify para.ini (using sed), run this again,
#               then add the sub-images to "subImages" folder
#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 28 October, 2018


# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

para_file=$1
if [ ! -f $para_file ]; then
   echo "File ${para_file} not exists in current folder: ${PWD}"
   exit 1
fi

deeplabRS=~/codes/PycharmProjects/DeeplabforRS
para_py=${deeplabRS}/parameters.py


train_shp=$(python2 ${para_py} -p ${para_file} training_polygons_sub )

label_raster=$(python2 ${para_py} -p ${para_file} input_label_image )
input_image=$(python2 ${para_py} -p ${para_file} input_image_path)

subImage_dir=$(python2 ${para_py} -p ${para_file} input_train_dir)
subLabel_dir=$(python2 ${para_py} -p ${para_file} input_label_dir)


# since there could be multiple grount truth images, then we should add files to this folder instead of deleting them
mkdir -p ${subImage_dir}
mkdir -p ${subLabel_dir}


###pre-process UAV images, extract the training data from the whole image
dstnodata=$(python2 ${para_py} -p ${para_file} dst_nodata)
buffersize=$(python2 ${para_py} -p ${para_file} buffer_size)
rectangle_ext=$(python2 ${para_py} -p ${para_file} b_use_rectangle)

${deeplabRS}/extract_target_imgs.py -n ${dstnodata} -b ${buffersize} ${rectangle_ext} $train_shp ${input_image}  -o ${subImage_dir}
${deeplabRS}/extract_target_imgs.py -n ${dstnodata} -b ${buffersize} ${rectangle_ext} $train_shp ${label_raster}  -o ${subLabel_dir}
