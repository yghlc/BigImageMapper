#!/usr/bin/env bash

# draw  precision-recall curve for all the test

# run this script on Cryo03
# in /home/hlc/Data/Qinghai-Tibet/beiluhe/beiluhe_planet/autoMapping/BLH_basin_deeplabV3+_1

# Exit immediately if a command exits with a non-zero status.
set -e

eo_dir=~/codes/PycharmProjects/Landuse_DL
#cd ${eo_dir}
#git pull
#cd -

deeplabRS=~/codes/PycharmProjects/DeeplabforRS

#cd /home/hlc/Data/Qinghai-Tibet/beiluhe/beiluhe_planet/autoMapping/BLH_basin_deeplabV3+_1
# copy this script to /home/hlc/Data/Qinghai-Tibet/beiluhe/beiluhe_planet/autoMapping/BLH_basin_deeplabV3+_1

echo "$(date): start plotting k-fold precision recall curves ">>"plot_p_r_curves.log"

# draw k-fold test
for k in  3 5 10 ; do
    for test in {1..5}; do

    echo $k $test >> "plot_p_r_curves.log"
    ${eo_dir}/thawslumpScripts/test_kfold_figure.sh $k $test
    mv accuracies_log.txt result_backup/${k}fold_test${test}/.

    done
done

echo "$(date): finish plotting k-fold precision recall curves ">>"plot_p_r_curves.log"
##########################################################################################
##########
cd result_backup/img_aug_test_results
# draw results of image augmentation
#img_aug_test_results

rm accuracies_log.txt | true
# plot Precision-recall curve
# get the first ini file,  others should have the same validate polgons
para_file=../../para.ini

echo "$(date): start plotting image augmentation experiment precision recall curves ">>"plot_p_r_curves.log"
output=p_r_img_augmentation_new.jpg
shp_list=$(ls BLH_basin_deeplabV3+_*_exp10_iter30000_post_imgAug*.shp | grep -v _TP )
python ${deeplabRS}/plot_accuracies.py -p ${para_file} ${shp_list} -o ${output}
mv accuracies_log.txt accuracies_bak.txt
echo "$(date): finish plotting image augmentation experiment precision recall curves ">>"plot_p_r_curves.log"

echo "$(date): start plotting image augmentation experiment (no post processing) precision recall curves ">>"plot_p_r_curves.log"
output=p_r_img_augmentation_noPost.jpg
shp_list=$(ls BLH_basin_deeplabV3+_*_exp10_iter30000_imgAug*.shp | grep -v _TP )
python ${deeplabRS}/plot_accuracies.py -p ${para_file} ${shp_list} -o ${output}
mv accuracies_log.txt accuracies_log_noPost.txt
echo "$(date): finish plotting image augmentation experiment (no post processing) precision recall curves ">>"plot_p_r_curves.log"

#plot top 5 ap
echo "$(date): start plotting image augmentation experiment (top5) precision recall curves ">>"plot_p_r_curves.log"
output=p_r_img_augmentation_top5.jpg
shp_list=$(ls *_post_imgAug1.shp *_post_imgAug16.shp *_post_imgAug22.shp *_post_imgAug26.shp *_post_imgAug30.shp )
python ${deeplabRS}/plot_accuracies.py -p ${para_file} ${shp_list} -o ${output}
mv accuracies_log.txt accuracies_log_top5.txt
echo "$(date): finish plotting image augmentation experiment (top5) precision recall curves ">>"plot_p_r_curves.log"

cd -

################################################################
# draw results of the first test
cd result_backup/first_test
echo "$(date): start plotting first test precision recall curves ">>"plot_p_r_curves.log"
output=p_r_first_test.jpg
shp_list=$(ls BLH_basin_deeplabV3+_1_exp10_iter30000*_2.shp)
python ${deeplabRS}/plot_accuracies.py -p ${para_file} ${shp_list} -o ${output}
echo "$(date): finish plotting first test precision recall curves ">>"plot_p_r_curves.log"
cd -

################################################################
# draw results of other bands
cd result_backup/other_bands_test_blur_crop_scale
echo "$(date): start plotting other bands precision recall curves ">>"plot_p_r_curves.log"
output=p_r_other_bands.jpg
shp_list=$(ls *_post_b1_ndvi_ndwi.shp *_post_pca3b_3.shp )
python ${deeplabRS}/plot_accuracies.py -p ${para_file} ${shp_list} -o ${output}
echo "$(date): finish plotting other bands precision recall curves ">>"plot_p_r_curves.log"
cd -

