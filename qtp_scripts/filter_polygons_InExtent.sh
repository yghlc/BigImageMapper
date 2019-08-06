#!/bin/bash

#introduction: filter polygons if they are in the extent
# run this script in

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 6 August, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

SECONDS=0

data_dir=~/Data/Qinghai-Tibet/entire_QTP_images/sentinel-2/autoMapping/BLH_deeplabV3+_1/result_backup

input_shp=${data_dir}/BLH_deeplabV3+_1_exp1_iter30000_prj_post.shp

# usually, this shape file should only contains one simple polygon.
#  biggest_permafrost_area_from_zou
#extent_shp=~/Data/Qinghai-Tibet/biggest_permafrost_area_from_zou/extent/biggest_permafrost_area_from_zou_outline_prj.shp

extent_shp=~/Data/Qinghai-Tibet/Qinghai-Tibet_Plateau_shp/Qinghai-Tibet_Plateau_outline2_gee/QTP_outline_simplified_0.1_Albers.shp

save_shp=${data_dir}/BLH_deeplabV3+_1_exp1_iter30000_prj_post_crop.shp

# make sure the two shape files below has the same projection info (coordinate reference system, CRS)
crs_shp=$(gdalsrsinfo -o EPSG ${input_shp})
crs_ext=$(gdalsrsinfo -o EPSG ${extent_shp})
echo $crs_shp
echo $crs_ext

if [ "$crs_shp" == "$crs_ext" ]; then
  echo "The coordinate reference system (CRS) of two input is the same "
else
  echo "Error, The coordinate reference system (CRS) of two input is the different, terminate "
  exit 1
fi

#ogr2ogr -t_srs  ${t_srs}  I${n}_${testid}_prj.shp I${n}_${testid}.shp


###############################################################
# a simply way is to clip the input_shp in QGIS (vector-> Geoprocessing tools-> clip) or GDAL (ogr2ogr)

# -progress: Only works if input layers have the “fast feature count” capability.
ogr2ogr -progress -clipsrc ${extent_shp} ${save_shp} ${input_shp}



duration=$SECONDS
#echo "$(date): The time of filtering polygons based on extent : ${duration} seconds">>"time_cost.txt"
