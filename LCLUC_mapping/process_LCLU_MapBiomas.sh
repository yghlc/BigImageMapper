#!/bin/bash

# Introduction: merge LCLU map from https://mapbiomas.org/en

# run this script in ~/Bhaltos2/lingcaoHuang/LandCover_LandUse_Change/LCLUC_MapBiomas_Gabriel

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

dir=~/Data/LandCover_LandUse_Change/LCLUC_MapBiomas_Gabriel

nodata=0
# UTM zone 22S
t_prj=EPSG:32722

crop_shp=../SouthBrazil/SouthBrazil_RS.shp

for year in $(seq 1985 2019); do
  echo $year

  # these two *-2019.tif is large and require a lot of memory, need to run on Tesia

  gdal_merge.py -o COLECAO_5_DOWNLOADS_COLECOES_ANUAL_${year}_merge.tif -n ${nodata} -a_nodata ${nodata} -co compress=lzw  \
          PAMPA_1985-2019/*-${year}.tif   MATAATLANTICA_1985-2019/*-${year}.tif

  # reprojection
  gdalwarp -t_srs ${t_prj} -co compress=lzw  COLECAO_5_DOWNLOADS_COLECOES_ANUAL_${year}_merge.tif \
          COLECAO_5_DOWNLOADS_COLECOES_ANUAL_${year}_merge_prj.tif

  # crop
  gdalwarp -co compress=lzw  -cutline ${crop_shp} -crop_to_cutline  \
  COLECAO_5_DOWNLOADS_COLECOES_ANUAL_${year}_merge_prj.tif COLECAO_5_DOWNLOADS_COLECOES_ANUAL_${year}_merge_prj_crop.tif

done




