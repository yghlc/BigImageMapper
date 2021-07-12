#!/usr/bin/env bash

# plot land cover and land use map in south Brazil from MapBiomas for certain regions
# projection: EPSG:32722 - WGS 84 / UTM zone 22S

# note:
# if we want to apply color table, we should save the raster to NetCDF format, otherwise, colorTable cannot work.

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 11 July, 2021

img_dir=~/Data/LandCover_LandUse_Change/LCLUC_MapBiomas_Gabriel
#shp_dir=~/Data/LandCover_LandUse_Change/Potential_areas_for_preliminary_study/areas_extent

## for area1
#xmin=256854
#xmax=270842
#ymin=6836773
#ymax=6849079


# for dam1_surrounding
xmin=-3756
xmax=18132
ymin=6713740
ymax=6731993


## for dam2_surrounding
#xmin=-25190
#xmax=-126
#ymin=6634467
#ymax=6654893


function crop_img(){
    in=$1
    out=$2
    # crop
    gdal_translate  -of NetCDF -projwin ${xmin} ${ymax} ${xmax} ${ymin} $in ${out}
}

# make a color table, then modify it
# #gmt makecpt -Cbatlow -T1/41/1 -D > color.cpt

#width=$(expr ${xmax} - ${xmin})
width=12c  # 5 cm
echo ${width}

function plot_LCLU_map(){
  img=$1
  year=$2
  area_n=$3

  # crop
  crop_img $img ${year}.nc

  # -V Select verbose mode, d for debug  -Vd
  gmt begin LCLU_map_${year}_${area_n} jpg
    # multiple plot # 1 row by 3 col,
    # -M margin
    # -F size
    # -R${xmin}/${xmax}/${ymin}/${ymax}
    # -A autolable
    gmt basemap -Blrbt -JX${width}  -R${xmin}/${xmax}/${ymin}/${ymax}  #  -JU22S/${width} ${extlatlon}
#    gmt makecpt -Cbatlow -T1/41/1 -D -Z  #- > color.cpt
    gmt grdimage ${year}.nc -CcolorMapBiomas.cpt

    # add scale bar
    gmt basemap -Ln0.8/0.12+w2000+lm --FONT_ANNOT_PRIMARY=10p,Helvetica,red --MAP_SCALE_HEIGHT=5p --MAP_TICK_PEN_PRIMARY=2p,red

    # add legend
#    gmt legend -Dn0.9/0.12  #${year}.nc

    # draw colorbar
    gmt colorbar -CcolorMapBiomas.cpt -B


  gmt end #show

}

for year in $(seq 1985 2019); do
  echo $year
  map_tif=${img_dir}/COLECAO_5_DOWNLOADS_COLECOES_ANUAL_${year}_merge_prj_crop.tif

#  plot_LCLU_map ${map_tif} $year area1
  plot_LCLU_map ${map_tif} $year dam1_surr
#  plot_LCLU_map ${map_tif} $year dam2_surr


done

#year=2019
#map_tif=${img_dir}/COLECAO_5_DOWNLOADS_COLECOES_ANUAL_${year}_merge_prj_crop.tif
#plot_LCLU_map ${map_tif} $year area1


rm *.nc


