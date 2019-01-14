#!/bin/bash

# convert to tif format
# run in ~/Data/Qinghai-Tibet/beiluhe/beiluhe_spot/georeference_tif

outdir=../georeference_tif
mkdir ${outdir}
for dat in $(ls *.dat); do

	echo $dat
	filename=$(basename "$dat")
	filename_no_ext="${filename%.*}"
	gdal_translate -of GTiff $dat ${outdir}/${filename_no_ext}.tif

done
