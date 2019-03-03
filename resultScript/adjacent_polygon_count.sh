#!/usr/bin/env bash

# conduct the analysis of the relation between IoU and the count of adjacent ground truths

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 1 March, 2019

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

code_dir=~/codes/PycharmProjects/Landuse_DL
# folder contains results
res_dir=~/Data/Qinghai-Tibet/beiluhe/result/result_paper_mapping_RTS_dl_beiluhe

### polygons
# the 202 ground truth polygons
ground_truth=${res_dir}/identified_ThawSlumps_prj_post.shp

# polyon without post-processing (removing polygons based on their areas)
# only one ground truth is missing
#shp_imgAug16_NOpost=${res_dir}/img_aug_test_results/BLH_basin_deeplabV3+_1_exp9_iter30000_imgAug16.shp
## no ground truth is missing
#shp_imgAug17_NOpost=${res_dir}/img_aug_test_results/BLH_basin_deeplabV3+_1_exp9_iter30000_imgAug17.shp

# only one ground truth is missing
shp_imgAug16_NOpost_tp=${res_dir}/img_aug_test_results/BLH_basin_deeplabV3+_1_exp9_iter30000_imgAug16_TP.shp
# no ground truth is missing
shp_imgAug17_NOpost_tp=${res_dir}/img_aug_test_results/BLH_basin_deeplabV3+_1_exp9_iter30000_imgAug17_TP.shp

# only one ground truth is missing
shp_imgAug16_NOpost_tp_intersec1=${res_dir}/img_aug_test_results/BLH_basin_deeplabV3+_1_exp9_iter30000_imgAug16_TP_intersec1.shp
shp_imgAug16_NOpost_tp_intersec2p=${res_dir}/img_aug_test_results/BLH_basin_deeplabV3+_1_exp9_iter30000_imgAug16_TP_intersec2p.shp
# no ground truth is missing
shp_imgAug17_NOpost_tp_intersec1=${res_dir}/img_aug_test_results/BLH_basin_deeplabV3+_1_exp9_iter30000_imgAug17_TP_intersec1.shp

# step 1: get the count of adjacent polygons based on ground_truths
#${code_dir}/resultScript/add_info2Pylygons.py ${ground_truth}  -n "adj_count" -b 300

# step 2: get new polygons by interest ground truths with mapped ones (e.g., shp_imgAug17_NOpost or shp_imgAug16_NOpost)
#${code_dir}/resultScript/get_intersection.py ${ground_truth} ${shp_imgAug17_NOpost_tp} \
# -o ${res_dir}/intersect_ground_truth_imgAug17_NOpost_tp.shp -c "adj_count"
#
#${code_dir}/resultScript/get_intersection.py ${ground_truth} ${shp_imgAug16_NOpost_tp} \
# -o ${res_dir}/intersect_ground_truth_imgAug16_NOpost_tp.shp -c "adj_count"

# also step 2: by with different method: only keep the mapped polygons intersect with one ground truth
# need modify to get ${shp_imgAug16_NOpost_tp_intersec2p}, which only keep the polygons intersect with 2+ ground truths
#${code_dir}/resultScript/remove_polyongs.py ${shp_imgAug16_NOpost_tp} -o ${shp_imgAug16_NOpost_tp_intersec2p} \
#    -v ${ground_truth}  -c "adj_count"

${code_dir}/resultScript/remove_polyongs.py ${shp_imgAug16_NOpost_tp} -o ${shp_imgAug16_NOpost_tp_intersec1} \
    -v ${ground_truth}  -c "adj_count"

${code_dir}/resultScript/remove_polyongs.py ${shp_imgAug17_NOpost_tp} -o ${shp_imgAug17_NOpost_tp_intersec1} \
    -v ${ground_truth}  -c "adj_count"

# step 3: calculate the IOU values
#${code_dir}/resultScript/add_info2Pylygons.py ${res_dir}/intersect_ground_truth_imgAug17_NOpost_tp.shp -v ${ground_truth} -n "IoU"
#
#${code_dir}/resultScript/add_info2Pylygons.py ${res_dir}/intersect_ground_truth_imgAug16_NOpost_tp.shp -v ${ground_truth} -n "IoU"


# step 4: plot scatter figure
# in script "plot_scatter.py"







