#!/usr/bin/env bash

# rasterize mapping polygons

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 29 September, 2021

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

merge_py=~/codes/PycharmProjects/Landuse_DL/datasets/merge_shapefiles.py
ras_py=~/codes/PycharmProjects/Landuse_DL/datasets/rasterize_polygons.py

# Houston region
# US National Atlas Equal Area, do we need to reproject?
t_prj=EPSG:2163


shp_dir=~/Data/flooding_area/Houston/Harvey_shp_files/Shp
out_dir=~/Data/flooding_area/flooding_rasters_other_results/Houston
shp1=${shp_dir}/TwoweekMODIS20170904a_region.shp
shp2=${shp_dir}/TwoweekMODIS20170904b_region.shp

mkdir -p ${out_dir}

# merge the two shapefiles
out_name=TwoweekMODIS20170904_region
merge_out=${out_dir}/${out_name}.shp
if [ ! -f ${merge_out} ]; then
  ${merge_py} ${shp1} ${shp2} -o ${merge_out}
else
   echo "file exist:" ${merge_out}
fi

# reproject
shp_prj=${out_dir}/${out_name}_prj.shp
if [ ! -f ${shp_prj} ]; then
  ogr2ogr -t_srs ${t_prj} ${shp_prj} ${merge_out}
else
  echo "file exist:" ${shp_prj}
fi

# to raster
ref_raster=~/Data/flooding_area/Houston/Houston_mosaic/S1A_IW_GRDH_1SDV_20170829_Sigma0_VH_Ptf_binaryLM_prj_255.tif
${ras_py} -r ${ref_raster} -o ${out_dir} -b 1 ${shp_prj}

# compress the raster
gdal_translate -co "compress=lzw" ${out_dir}/${out_name}_prj_label.tif ${out_dir}/tmp.tif
rm ${out_dir}/${out_name}_prj_label.tif
mv ${out_dir}/tmp.tif ${out_dir}/${out_name}_prj_label.tif






