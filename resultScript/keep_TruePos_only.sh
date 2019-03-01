#!/usr/bin/env bash

# eep the true positive only, i.e., remove polygons with IOU less than or equal to 0.5.

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 26 February, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

code_dir=~/codes/PycharmProjects/Landuse_DL
# folder contains results
res_dir=~/Data/Qinghai-Tibet/beiluhe/result/result_paper_mapping_RTS_dl_beiluhe

### polygons
# mapped
shp_pre=${res_dir}/img_aug_test_results/BLH_basin_deeplabV3+_1_exp9_iter30000_post_imgAug16

mainline_pre=${res_dir}/identified_ThawSlumps_MaiinLines_prj

# remove False positive
#${code_dir}/resultScript/remove_polyongs.py ${shp_pre}.shp -o ${shp_pre}_TP.shp -f 'IoU' \
#    -t 0.5 -l ${mainline_pre}.shp --output_mainline=${mainline_pre}_TP.shp --bsmaller



# polyon without post-processing (removing polygons based on their areas)
shp_imgAug16_NOpost_pre=${res_dir}/img_aug_test_results/BLH_basin_deeplabV3+_1_exp9_iter30000_imgAug16
shp_imgAug17_NOpost_pre=${res_dir}/img_aug_test_results/BLH_basin_deeplabV3+_1_exp9_iter30000_imgAug17

# remove IoU value equal to 0
#${code_dir}/resultScript/remove_polyongs.py ${shp_imgAug16_NOpost_pre}.shp -o ${shp_imgAug16_NOpost_pre}_TP.shp -f 'IoU' \
#    -t 0.00001  --bsmaller
${code_dir}/resultScript/remove_polyongs.py ${shp_imgAug17_NOpost_pre}.shp -o ${shp_imgAug17_NOpost_pre}_TP.shp -f 'IoU' \
    -t 0.00001  --bsmaller

