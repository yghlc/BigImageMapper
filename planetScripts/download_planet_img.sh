#!/usr/bin/env bash

## Introduction:  Merge NDVI, NDWI, and one RGB band to a three bands image.

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 5 October, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

eo_dir=~/codes/PycharmProjects/Landuse_DL

# on Cryo06, to gdalsrsinfro (>2.3) and python 3
export PATH=~/programs/anaconda3/bin:$PATH

shp_file=~/Data/Qinghai-Tibet/qtp_thaw_slumps/rts_polygons_s2_2018/qtp_train_polygons_s2_2018_v2.shp
save_folder=planet_sr_images

start_date=2018-05-20
end_date=2018-06-01
cloud_cover_thr=0.3
item_type=PSScene4Band
#account=huanglingcao@link.cuhk.edu.hk
account=liulin@cuhk.edu.hk

${eo_dir}/planetScripts/download_planet_img.py ${shp_file} ${save_folder} \
-s ${start_date} -e ${end_date} -c ${cloud_cover_thr} -i ${item_type} -a ${account}
