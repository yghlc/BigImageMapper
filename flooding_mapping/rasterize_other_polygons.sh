#!/usr/bin/env bash

# rasterize mapping polygons

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 4 July, 2021

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

merge_py=~/codes/PycharmProjects/Landuse_DL/datasets/merge_shapefiles.py
ras_py=~/codes/PycharmProjects/Landuse_DL/datasets/rasterize_polygons.py

# Houston region
# US National Atlas Equal Area, do we need to reproject?
t_prj=EPSG:2163


shp_dir=~/Data/flooding_area/Houston/Harvey_shp_files/Shp
out_dir=~/Data/flooding_area/flooding_rasters_other_results/Houston
shp1=${shp_dir}/2017082920170805S1acleaned_region.shp
shp2=${shp_dir}/2017082920170805S1bcleaned_region.shp

mkdir -p ${out_dir}

# merge the two shapefiles
merge_out=${out_dir}/2017082920170805_cleaned_region.shp
if [ ! -f ${merge_out} ]; then
  ${merge_py} ${shp1} ${shp2} -o ${merge_out}
else
   echo "file exist:" ${merge_out}
fi

# reproject
shp_prj=${out_dir}/2017082920170805_cleaned_region_prj.shp
if [ ! -f ${shp_prj} ]; then
  ogr2ogr -t_srs ${t_prj} ${shp_prj} ${merge_out}
else
  echo "file exist:" ${shp_prj}
fi

## to raster
#ref_raster=~/Data/flooding_area/Houston/Houston_mosaic/S1A_IW_GRDH_1SDV_20170829_Sigma0_VH_Ptf_binaryLM_prj_255.tif
#${ras_py} -r ${ref_raster} -o ${out_dir} -b 1 ${shp_prj}


#####################################################################################################
# Goalpara region (UTM 46N)
t_prj=EPSG:32646

shp_dir=~/Data/flooding_area/Goalpara/Goapara_shp/shp
out_dir=~/Data/flooding_area/flooding_rasters_other_results/Goapara
shp1=${shp_dir}/20200705MODIS_region.shp

mkdir -p ${out_dir}

# reproject
shp_prj=${out_dir}/20200705MODIS_region_prj.shp
if [ ! -f ${shp_prj} ]; then
  ogr2ogr -t_srs ${t_prj} ${shp_prj} ${shp1}
else
  echo "file exist:" ${shp_prj}
fi

# to raster
ref_raster_dir=~/Data/flooding_area/Goalpara/Goalpara_power_transform_prj_8bit
ref_raster=${ref_raster_dir}/S1A_IW_GRDH_1SDV_20200703T235524_20200703T235549_033297_03DB97_AAB7_Sigma0_VH_Ptf_common_grid_cut_prj_8bit.tif
${ras_py} -r ${ref_raster} -o ${out_dir} -b 1 ${shp_prj}














