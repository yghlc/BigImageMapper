#!/usr/bin/env bash

# rasterize mapping polygons

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 3 July, 2021

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

shp_dir=~/Data/flooding_area/automapping/houston_deeplabV3+_1/result_backup
out_dir=~/Data/flooding_area/mapping_polygons_rasters

py=~/codes/PycharmProjects/Landuse_DL/datasets/rasterize_polygons.py

#for shp in $(ls ${shp_dir}/*/*/*.shp); do
#  echo $shp
#done
function rasterize_map() {
  folder=$1
  ref_tif_dir=$2
  for shp in $(ls ${shp_dir}/${folder}/*/*.shp); do
    echo $shp
    save_dir=${out_dir}/$folder
    name=$(basename $shp)
    date_str=$(echo $name | cut -d "_"  -f 3)

    ref_raster=$(ls ${ref_tif_dir}/*${date_str}*.tif)
    echo $ref_raster
    if [ ! -f ${ref_raster} ]; then
      echo "error, ${ref_raster} not exist, skip"
      continue
    fi
    mkdir -p  ${save_dir}
    ${py} -r ${ref_raster} -o ${save_dir} -b 1 ${shp}
  done
}
area_exp=exp1_grd_Goalpara
ref_tif_dir=~/Data/flooding_area/Goalpara/Goalpara_power_transform_prj_8bit
rasterize_map ${area_exp} ${ref_tif_dir}




