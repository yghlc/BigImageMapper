#!/usr/bin/env bash

# eep the true positive only, i.e., remove polygons with IOU less than or equal to 0.5.

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 26 February, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

code_dir=~/codes/PycharmProjects/Landuse_DL
# folder contains results
#res_dir=~/Data/Qinghai-Tibet/beiluhe/result/result_paper_mapping_RTS_dl_beiluhe
res_dir=~/Data/Qinghai-Tibet/beiluhe/result/result_paper_mapping_RTS_dl_beiluhe_revised2019

### polygons
# mapped
#shp_pre=${res_dir}/img_aug_test_results/BLH_basin_deeplabV3+_1_exp9_iter30000_post_imgAug16
shp_pre=${res_dir}/img_aug_test_results/BLH_basin_deeplabV3+_1_b_exp10_iter30000_post_imgAug22

#mainline_pre=${res_dir}/identified_ThawSlumps_MaiinLines_prj
mainline_pre=${res_dir}/identified_ThawSlumps_MaiinLines_utm


# remove False positive
#note that, after remove, the number of main lines are greater than the number of polygons in "shp_polygon"
#This is because, in Beiluhe, some mapped thaw slumps close to each other were merged to one
#${code_dir}/resultScript/remove_polyongs.py ${shp_pre}.shp -o ${shp_pre}_TP.shp -f 'IoU' \
#    -t 0.5 -l ${mainline_pre}.shp --output_mainline=${mainline_pre}_TP.shp --bsmaller



# polyon without post-processing (removing polygons based on their areas)
#shp_imgAug16_NOpost_pre=${res_dir}/img_aug_test_results/BLH_basin_deeplabV3+_1_exp9_iter30000_imgAug16
#shp_imgAug17_NOpost_pre=${res_dir}/img_aug_test_results/BLH_basin_deeplabV3+_1_exp9_iter30000_imgAug17

# remove IoU value equal to 0
#${code_dir}/resultScript/remove_polyongs.py ${shp_imgAug16_NOpost_pre}.shp -o ${shp_imgAug16_NOpost_pre}_TP.shp -f 'IoU' \
#    -t 0.00001  --bsmaller
#${code_dir}/resultScript/remove_polyongs.py ${shp_imgAug17_NOpost_pre}.shp -o ${shp_imgAug17_NOpost_pre}_TP.shp -f 'IoU' \
#    -t 0.00001  --bsmaller


# polyon without post-processing (removing polygons based on their areas)
shp_imgAug22_NOpost_pre=${res_dir}/img_aug_test_results/BLH_basin_deeplabV3+_1_b_exp10_iter30000_imgAug22
# remove IoU value equal to 0
${code_dir}/resultScript/remove_polyongs.py ${shp_imgAug22_NOpost_pre}.shp -o ${shp_imgAug22_NOpost_pre}_TP.shp -f 'IoU' \
    -t 0.00001  --bsmaller
