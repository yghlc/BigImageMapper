#!/bin/bash

export PATH=/usr/bin:$PATH

para_file=pre_para.ini
para_py=/home/hlc/codes/PycharmProjects/DeeplabforRS/parameters.py

eo_dir=$(python2 ${para_py} -p ${para_file} codes_dir)

shp_file=$(python2 ${para_py} -p ${para_file} training_polygons) 
out_raster=$(python2 ${para_py} -p ${para_file} output_label_image )

rm ${out_raster}

${eo_dir}/prepare_raster.py -p ${para_file} ${shp_file} ${out_raster} 

#default nodata in output is 255, so set it the 0
#gdal_translate -a_nodata 254 ${out_raster}  temp.tif
#gdal_calc.py -A temp.tif  --outfile=${out_raster} --calc="A==1"  --debug --type='Byte' --overwrite
#rm temp.tif

rm *_AllClass.tif
rm *_oneClass.tif
