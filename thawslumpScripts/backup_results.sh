#!/bin/bash

#introduction: Copy the results of mapping thaw slumps base on DeeplabV3+
#
#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 27 October, 2018

para_file=$1
if [ ! -f $para_file ]; then
   echo "File ${para_file} not exists in current folder: ${PWD}"
   exit 1
fi

test=$2


para_py=~/codes/PycharmProjects/DeeplabforRS/parameters.py
NUM_ITERATIONS=$(python2 ${para_py} -p ${para_file} export_iteration_num)
expr_name=$(python2 ${para_py} -p ${para_file} expr_name)
trail=iter${NUM_ITERATIONS}

testid=$(basename $PWD)_${expr_name}_${trail}
inf_dir=inf_results

mkdir -p result_backup

cp_shapefile ${inf_dir}/${testid}_post result_backup/${testid}_post_${test}
#cp_shapefile ${inf_dir}/${testid}_merged result_backup/${testid}_merged_${test}
cp_shapefile ${inf_dir}/${testid} result_backup/${testid}_${test}

cp ${inf_dir}/${para_file} result_backup/${testid}_para_${test}.ini
cp ${inf_dir}/evaluation_report.txt result_backup/${testid}_eva_report_${test}.txt
cp otb_acc_log.txt  result_backup/${testid}_otb_acc_${test}.txt

echo "complete: copy result files to result_backup, expriment: $expr_name, iterations: $NUM_ITERATIONS & copyNumber: _$test"