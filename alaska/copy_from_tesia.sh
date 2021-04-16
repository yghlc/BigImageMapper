#!/bin/bash

dir=/home/lihu9680/Data/temp/alaskaNS_yolov4_1/multi_inf_results/alaska_north_slope_hillshade_2010to2017

for id in 1245 2078 2018 2200 2 3171 900 2104 1199 1179; do
    img=I${id}
    echo $img
    mkdir ${img}
    scp -r ${tesia_host}:${dir}/${id}.txt .
    cd ${img}
    scp $tesia_host:${dir}/${img}/${img}_* .

    tif=$(cat ../${id}.txt)
    scp $tesia_host:${tif} .

    cd ..
done


