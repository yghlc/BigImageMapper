#!/usr/bin/env bash

# rasterize mapping polygons

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 19 October, 2021

# Exit immediately if a command exits with a non-zero status. E: error trace
#set -eE -o functrace

shp_dir=~/Data/flooding_area/Houston/houston_train_polygons
save_dir=~/Data/flooding_area/train_polygons_rasters

py=~/codes/PycharmProjects/Landuse_DL/datasets/rasterize_polygons.py

shp=${shp_dir}/houston_flood_polygons_20170829.shp

name_noext=houston_flood_polygons_20170829

mkdir -p  ${save_dir}
res=10

echo ${py}
${py} --pixel_size_x=${res} --pixel_size_y=${res} -o ${save_dir} -b 1 ${shp}


## compress the raster
gdal_translate -co "compress=lzw" ${save_dir}/${name_noext}_label.tif ${save_dir}/tmp.tif
rm ${save_dir}/${name_noext}_label.tif
mv ${save_dir}/tmp.tif ${save_dir}/${name_noext}_label.tif









