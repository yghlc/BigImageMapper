#!/bin/bash

#introduction: perform inference and post processing
#
#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 26 October, 2018

# Exit immediately if a command exits with a non-zero status.
set -e

# input a parameter: the path of para_file (e.g., para.ini)
para_file=$1
if [ ! -f $para_file ]; then
   echo "File ${para_file} not exists in current folder: ${PWD}"
   exit 1
fi

eo_dir=/home/hlc/codes/PycharmProjects/Landuse_DL
deeplabRS=~/codes/PycharmProjects/DeeplabforRS

para_py=~/codes/PycharmProjects/DeeplabforRS/parameters.py
NUM_ITERATIONS=$(python2 ${para_py} -p ${para_file} export_iteration_num)
expr_name=$(python2 ${para_py} -p ${para_file} expr_name)

trail=iter${NUM_ITERATIONS}
frozen_graph=frozen_inference_graph_${trail}.pb

testid=$(basename $PWD)_${expr_name}_${trail}
output=${testid}.tif
inf_dir=inf_${testid}


SECONDS=0


para_file=para.ini
para_py=~/codes/PycharmProjects/DeeplabforRS/parameters.py

#rm -r ${inf_dir}

python ${eo_dir}/grss_data_fusion/deeplab_inference.py --frozen_graph=${frozen_graph} --inf_output_dir=${inf_dir}

duration=$SECONDS
echo "$(date): time cost of inference: ${duration} seconds">>"time_cost.txt"
$SECONDS=0

# merge patches
### post processing
cd ${inf_dir}

    #python ${eo_dir}/gdal_class_mosaic.py -o ${output} -init 0 *_pred.tif
    gdal_merge.py -init 0 -n 0 -a_nodata 0 -o ${output} *_pred.tif
    #mv ${output} ../.

    gdal_polygonize.py -8 ${output} -b 1 -f "ESRI Shapefile" ${testid}.shp

    # post processing of shapefile
    cp ../${para_file}  ${para_file}
    min_area=$(python2 ${para_py} -p ${para_file} minimum_gully_area)
    min_p_a_r=$(python2 ${para_py} -p ${para_file} minimum_ratio_perimeter_area)
    ${deeplabRS}/polygon_post_process.py -p ${para_file} -a ${min_area} -r ${min_p_a_r} ${testid}.shp ${testid}_post.shp

cd ..

duration=$SECONDS
echo "$(date): time cost of post processing: ${duration} seconds">>"time_cost.txt"
