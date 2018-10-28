#!/bin/bash

echo $(basename $0) : "Create the ground truth images for a shape file (containing training polygons)"
#introduction: Create the ground truth images for a shape file (containing training polygons)
#               if there are multiple files, then modify para.ini (using sed), run this one again
#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 28 October, 2018

# use python2, this is called in exe.sh, so don't necessary to set $PATH
### python 2 on Cryo06
#export PATH=/usr/bin:$PATH
### python2 on Cryo03
#export PATH=/home/hlc/programs/anaconda2/bin:$PATH


# Exit immediately if a command exits with a non-zero status.
set -e

para_file=$1
if [ ! -f $para_file ]; then
   echo "File ${para_file} not exists in current folder: ${PWD}"
   exit 1
fi

deeplabRS=~/codes/PycharmProjects/DeeplabforRS
para_py=${deeplabRS}/parameters.py


shp_file=$(python2 ${para_py} -p ${para_file} training_polygons) 
label_raster=$(python2 ${para_py} -p ${para_file} input_label_image )

if [  -f $label_raster ]; then
   rm ${label_raster}
fi


${deeplabRS}/prepare_raster.py -p ${para_file} ${shp_file} ${label_raster}

#default nodata in output is 255, so set it the 0
#gdal_translate -a_nodata 254 ${out_raster}  temp.tif
#gdal_calc.py -A temp.tif  --outfile=${out_raster} --calc="A==1"  --debug --type='Byte' --overwrite
#rm temp.tif

out_dir=$(dirname $label_raster)
echo $out_dir
cd $out_dir
    rm *_AllClass.tif
    rm *_oneClass.tif
cd -
