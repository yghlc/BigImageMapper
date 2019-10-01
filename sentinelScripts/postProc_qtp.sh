#!/bin/bash

#introduction: perform post processing
#
#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 26 October, 2018

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

# input a parameter: the path of para_file (e.g., para.ini)
para_file=$1
if [ ! -f $para_file ]; then
   echo "File ${para_file} not exists in current folder: ${PWD}"
   exit 1
fi

test=$2

eo_dir=~/codes/PycharmProjects/Landuse_DL
deeplabRS=~/codes/PycharmProjects/DeeplabforRS

para_py=~/codes/PycharmProjects/DeeplabforRS/parameters.py

expr_name=$(python2 ${para_py} -p ${para_file} expr_name)
NUM_ITERATIONS=$(python2 ${para_py} -p ${para_file} export_iteration_num)
trail=iter${NUM_ITERATIONS}

testid=$(basename $PWD)_${expr_name}_${trail}
output=${testid}.tif
inf_dir=multi_inf_results

SECONDS=0

# the number of images in the list for inference
num=$(cat inf_image_list.txt | wc -l)

# merge patches
### post processing
cd ${inf_dir}

    for (( n=0; n<${num}; n++ ));
    do

    cd I${n}

    #python ${eo_dir}/gdal_class_mosaic.py -o ${output} -init 0 *_pred.tif
    if [ ! -f I${n}_${output} ]; then
        gdal_merge.py -init 0 -n 0 -a_nodata 0 -o I${n}_${output} I0_*.tif
    fi

    #mv ${output} ../.
    if [ ! -f I${n}_${testid}.shp ]; then
        gdal_polygonize.py -8 I${n}_${output} -b 1 -f "ESRI Shapefile" I${n}_${testid}.shp
    fi

    # reproject the shapefile from "GEOGCS (WGS84)" to "Cartesian (XY) projection"
    # need to modify it if switch to other regions
    t_srs=$(python2 ${para_py} -p ${para_file} cartensian_prj)

    if [ ! -f I${n}_${testid}_prj.shp ]; then
        ogr2ogr -t_srs  ${t_srs}  I${n}_${testid}_prj.shp I${n}_${testid}.shp
    fi


    # post processing of shapefile
    cp ../../${para_file}  ${para_file}
    min_area=$(python2 ${para_py} -p ${para_file} minimum_gully_area)
    min_p_a_r=$(python2 ${para_py} -p ${para_file} minimum_ratio_perimeter_area)
    ${deeplabRS}/polygon_post_process.py -p ${para_file} -a ${min_area} -r ${min_p_a_r} I${n}_${testid}_prj.shp I${n}_${testid}_prj_post.shp

    cd -

    done

cd ..

duration=$SECONDS
echo "$(date): time cost of post processing: ${duration} seconds">>"time_cost.txt"

########################################
# copy results
#if [ ${num} -lt 2 ] ; then
#    bak_dir=result_backup
#    echo "the results contains only one shape file"
#else
#    bak_dir=result_backup/${testid}_${test}_tiles
#    echo "the results containes twor or more shape files"
#fi

bak_dir=result_backup/${testid}_${test}_tiles
mkdir -p ${bak_dir}
for (( n=0; n<${num}; n++ ));
    do

    cp_shapefile ${inf_dir}/I${n}/I${n}_${testid}_prj_post ${bak_dir}/I${n}_${testid}_prj_post_${test} | true
    cp_shapefile ${inf_dir}/I${n}/I${n}_${testid}_prj ${bak_dir}/I${n}_${testid}_prj_${test} | true

    cp ${para_file} result_backup/${testid}_para_${test}.ini
#    cp ${inf_dir}/evaluation_report.txt result_backup/${testid}_eva_report_${test}.txt
#    cp otb_acc_log.txt  result_backup/${testid}_otb_acc_${test}.txt

    echo "complete: copy result files to result_backup, expriment: $expr_name, iterations: $NUM_ITERATIONS & copyNumber: _$test"

done

########################################