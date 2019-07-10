#!/usr/bin/env bash

# set nodata for images
nodata=0

for tif in $(ls *.tif); do

    gdal_edit.py -a_nodata ${nodata} ${tif}
done

