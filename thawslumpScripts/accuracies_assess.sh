#!/bin/bash

#introduction: perform accuracy assessment
#
#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 1 January, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

# input a parameter: the path of para_file (e.g., para.ini)
para_file=$1
if [ ! -f $para_file ]; then
   echo "File ${para_file} not exists in current folder: ${PWD}"
   exit 1
fi

eo_dir=~/codes/PycharmProjects/Landuse_DL
deeplabRS=~/codes/PycharmProjects/DeeplabforRS

para_py=~/codes/PycharmProjects/DeeplabforRS/parameters.py

expr_name=$(python2 ${para_py} -p ${para_file} expr_name)
NUM_ITERATIONS=$(python2 ${para_py} -p ${para_file} export_iteration_num)
trail=iter${NUM_ITERATIONS}

testid=$(basename $PWD)_${expr_name}_${trail}
output=${testid}.tif
inf_dir=inf_results

SECONDS=0

# these two should have the same size
label_image=$(python2 ${para_py} -p ${para_file} input_label_image)
raster_mapped_results=${inf_dir}/${output}

# using sk-learn to calculate the total accuracies
${eo_dir}/grss_data_fusion/classify_assess.py ${label_image} ${raster_mapped_results}

# using otbcli_ComputeConfusionMatrix to get ConfusionMatrix, then compute total accruacies

otbcli_ComputeConfusionMatrix -progress -in ${raster_mapped_results} -out ConfusionMatrix.csv \
-ref raster -ref.raster.in ${label_image} -nodatalabel 255 -ram 2048


duration=$SECONDS
echo "$(date): time cost of accuracy assessment: ${duration} seconds">>"time_cost.txt"

