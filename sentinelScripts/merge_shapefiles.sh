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

test=$2

expr_name=$(python2 ${para_py} -p ${para_file} expr_name)
NUM_ITERATIONS=$(python2 ${para_py} -p ${para_file} export_iteration_num)
trail=iter${NUM_ITERATIONS}

testid=$(basename $PWD)_${expr_name}_${trail}

cd result_backup

tile_dir=${testid}_${test}_tiles

# crop a data set
function merge_shp() {
    out_shp=$1
    out_name=$2
    i=$3

    if [ -f "$out_shp" ]
    then
       echo "updating ${out_shp}"
       ogr2ogr -f "ESRI Shapefile" -update -append $out_shp $i -nln ${out_name}
    else
        echo "creating ${out_shp}"
        ogr2ogr -f "ESRI Shapefile" $out_shp $i
    fi
}


for first_shp in $(ls ${tile_dir}/I0_*.shp); do

#    out_name=${testid}_prj_post_${test}
    echo "First sub shape file:" $first_shp
    basename=$(basename $first_shp)
    # remove "I0_"
    out_shp=$(python -c "import sys; print('_'.join(sys.argv[1].split('_')[1:]))" $basename)
    out_name="${out_shp%.*}"
    if [ -f $out_shp ]; then
        echo "remove previous results"
        rm ${out_name}.*
    fi
    for i in $(ls ${tile_dir}/I*_${out_name}.shp)
    do
        echo "merging $i"
        merge_shp ${out_shp} ${out_name} ${i}
    done
    # convert to KML
    echo $"convert to KML format"
    ogr2ogr -f KML ${out_name}.kml ${out_shp}

done


#out_name=${testid}_prj_${test}
#out_shp=${out_name}.shp
#if [ -f $out_shp ]; then
#    echo "remove previous results"
#    rm ${out_name}.*
#fi
#for i in $(ls ${tile_dir}/*_prj*.shp | grep -v post)
#do
#    echo "merging $i"
#    merge_shp ${out_shp} ${out_name} ${i}
#done
## convert to KML
#echo $"convert to KML format"
#ogr2ogr -f KML ${out_name}.kml ${out_shp}

cd ..