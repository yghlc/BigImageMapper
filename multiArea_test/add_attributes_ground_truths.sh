#!/usr/bin/env bash

# add attributes to multiple ground truth polygons

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 17 January, 2021

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

deeplabRS=~/codes/PycharmProjects/DeeplabforRS


shps_txt=ground_truth_shp.txt

dir=~/Data/Arctic/canada_arctic/ground_truth_info


# input a parameter: the path of para_file (e.g., para.ini)
para_file=para.ini
if [ ! -f $para_file ]; then
   echo "File ${para_file} not exists in current folder: ${PWD}"
   exit 1
fi


while IFS= read -r line
do
    #echo "$line"
    input=${line}
    echo ${input}

    filename=$(basename ${input})
#    extension="${filename##*.}"
    filename_no_ext="${filename%.*}"

    output=${filename_no_ext}_post.shp

    ${deeplabRS}/polygon_post_process.py -p ${para_file} ${input} ${output}

done < "$shps_txt"



