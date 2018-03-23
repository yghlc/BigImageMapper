#!/usr/bin/env bash


eo_dir=/home/hlc/codes/PycharmProjects/Landuse_DL

result_map=$1
echo "assess on the subset of " $result_map

ground_truth=/home/hlc/Data/2018_IEEE_GRSS_Data_Fusion/2018_Release_Phase1/GT/Copy/2018_IEEE_GRSS_DFC_GT_TR.tif

#subset result map, because we only have a subset of ground truth
result_map_sub=${result_map}_sub

#ulx uly lrx lry
exent_of_GT=" 272056.000 3290290.000 274440.000 3289689.000"

# use gdal_translate with nearest resample instead of gdal_warp because this is a classification map

gdal_translate -r nearest -projwin ${exent_of_GT} ${result_map} ${result_map_sub}

#assess
${eo_dir}/classify_assess.py ${ground_truth} ${result_map_sub}