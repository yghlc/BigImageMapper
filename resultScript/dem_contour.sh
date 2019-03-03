#!/bin/bash

# get dem contour by using gdal_contour

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 3 March, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

dem_dir=~/Data/Qinghai-Tibet/beiluhe/DEM/srtm_30
outdir=${dem_dir}/dem_contour

mkdir -p ${outdir}

# the dem file
dem=${dem_dir}/beiluhe_srtm30_utm_basinExt.tif
output=${outdir}/beiluhe_srtm30_utm_basinExt_contour.shp

interval=100

rm ${output} | true
gdal_contour -b 1 -a strm30 -inodata -i ${interval} ${dem} ${output}



