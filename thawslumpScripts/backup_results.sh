#!/bin/bash

test=$1

expr=${PWD}
testid=$(basename $expr)
inf_dir=inf_results_exp2

cp_shapefile ${inf_dir}/${testid}_post result_backup/${testid}_post_test_${test}
cp_shapefile ${inf_dir}/${testid}_merged result_backup/${testid}_merged_test_${test}

cp ${inf_dir}/para.ini result_backup/para_post_test_${test}.ini 
cp ${inf_dir}/evaluation_report.txt result_backup/evaluation_report_post_test_${test}.txt
