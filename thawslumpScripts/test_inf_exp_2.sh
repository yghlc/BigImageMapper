#!/bin/bash

eo_dir=/home/hlc/codes/PycharmProjects/Landuse_DL
deeplabRS=~/codes/PycharmProjects/DeeplabforRS
cd ${eo_dir}
git pull
cd -

SECONDS=0
#set GPU on
export CUDA_VISIBLE_DEVICES=1

testid=$(basename $PWD)
inf_dir=inf_results_exp2
para_file=para.ini
para_py=~/codes/PycharmProjects/DeeplabforRS/parameters.py

#rm -r ${inf_dir}

python ${eo_dir}/grss_data_fusion/deeplab_inference.py --frozen_graph=frozen_inference_graph_iter60000.pb --inf_output_dir=${inf_dir}


# merge patches
### post processing
output=map_result_exp2.tif

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
echo "$(date): time cost of inference: ${duration} seconds">>"time_cost.txt"
