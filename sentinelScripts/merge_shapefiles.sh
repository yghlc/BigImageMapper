#!/usr/bin/env bash

#introduction: merge shape files (after post processing)
#
#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 21 July, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

# input a parameter: the path of para_file (e.g., para.ini)
para_file=$1
if [ ! -f $para_file ]; then
   echo "File ${para_file} not exists in current folder: ${PWD}"
   exit 1
fi
para_py=~/codes/PycharmProjects/DeeplabforRS/parameters.py

expr_name=$(python2 ${para_py} -p ${para_file} expr_name)
NUM_ITERATIONS=$(python2 ${para_py} -p ${para_file} export_iteration_num)
trail=iter${NUM_ITERATIONS}

testid=$(basename $PWD)_${expr_name}_${trail}

out_name=${testid}_prj_post
out_shp=${out_name}.shp

cd result_backup
cp ../${para_file} para.ini

for i in $(ls *_prj_post*.shp)
do

      if [ -f "$out_shp" ]
      then
           echo "updating ${out_shp}"
           ogr2ogr -f "ESRI Shapefile" -update -append $out_shp $i -nln ${out_name}
      else
            echo "creating ${out_shp}"
            ogr2ogr -f "ESRI Shapefile" $out_shp $i
#            exit
      fi

done

cd ..

# convert to KML
ogr2ogr -f KML ${out_name}.kml ${out_shp}
