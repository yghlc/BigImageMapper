#!/usr/bin/env bash

## Introduction: downloaded Planet images

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 5 October, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

word_dir=~/Data/Qinghai-Tibet/entire_QTP_images
eo_dir=~/codes/PycharmProjects/Landuse_DL

cd ${word_dir}

# on Cryo06, to gdalsrsinfro (>2.3) and python 3
export PATH=~/programs/anaconda3/bin:$PATH

#shp_file=~/Data/Qinghai-Tibet/qtp_thaw_slumps/rts_polygons_s2_2018/qtp_train_polygons_s2_2018_v2.shp
# mapping results
#shp_file=~/Data/Qinghai-Tibet/entire_QTP_images/sentinel-2/autoMapping/QTP_deeplabV3+_3/result_backup/QTP_deeplabV3+_3_exp2_iter30000_prj_post2_chpc_2_latlon.shp

# ground truths of thaw slumps in Beiluhe
#shp_file=~/Data/Qinghai-Tibet/beiluhe/thaw_slumps/train_polygons_for_planet_2018_revised_2019/identified_thawslumps_264_latlon.shp

# beiluhe river basin extent. (error: message is too long)
shp_file=~/Data/Qinghai-Tibet/beiluhe/beiluhe_reiver_basin_extent/beiluhe_reiver_basin_extent.shp

#save_folder=planet_sr_images

### download images of 2018
##start_date=2018-05-20
##end_date=2018-06-01
#start_date=2018-05-20
#end_date=2018-06-30
#save_folder=planet_sr_images/2018_May_Jun

## download images of 2018 July, August
#start_date=2018-07-01
#end_date=2018-08-31
#save_folder=planet_sr_images/2018_Jul_Aug

## download images of 2016
#start_date=2016-07-01
#end_date=2016-08-31
#save_folder=planet_sr_images/2016_Jul_Aug

## download images of 2017
#start_date=2017-07-01
#end_date=2017-08-31
#save_folder=planet_sr_images/2017_Jul_Aug

# download images of 2019
start_date=2019-07-01
end_date=2019-08-31
save_folder=planet_sr_images/2019_Jul_Aug


cloud_cover_thr=0.3
item_type=PSScene4Band
#account=huanglingcao@link.cuhk.edu.hk
account=liulin@cuhk.edu.hk

${eo_dir}/planetScripts/download_planet_img.py ${shp_file} ${save_folder} \
-s ${start_date} -e ${end_date} -c ${cloud_cover_thr} -i ${item_type} -a ${account}
