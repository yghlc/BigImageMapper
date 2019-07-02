#!/usr/bin/env bash

# run tests on using SVM classification
# copy this script to the working folder
# e.g., working folder: ~/Data/Qinghai-Tibet/beiluhe/beiluhe_landsat/auto_classification/BLH_landsattrend_svm_1
# then run

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace


# use python3 to use starmap function in Pool for parallel computing
export PATH=~/programs/anaconda3/bin:$PATH

# output log information to time_cost.txt

log='landsatTrend_svm_log.txt'
#rm ${log} || true   # or true: don't exit with error and can continue run

time_str=`date +%Y_%m_%d_%H_%M_%S`
SECONDS=0

echo ${time_str} >> ${log}

eo_dir=~/codes/PycharmProjects/Landuse_DL

#para_file=$1
para_file=para_landsat.ini
deeplabRS=~/codes/PycharmProjects/DeeplabforRS
para_py=${deeplabRS}/parameters.py
train_shp_all=$(python2 ${para_py} -p ${para_file} training_polygons )

input_image=$(python2 ${para_py} -p ${para_file} input_image_path )

##########pre-processing for SVM##########

rm "scaler_saved.pkl" || true
${eo_dir}/planetScripts/planet_svm_classify.py ${input_image}  -p
##
#
###########svm training##########

## remove previous subImages
#${eo_dir}/thawslumpScripts/remove_previous_data.sh ${para_file}
#
## get a ground truth raster if it did not exists or the corresponding shape file gets update
#${eo_dir}/thawslumpScripts/get_ground_truth_raster.sh ${para_file}
#
## get subImages (using four bands) and subLabels for training, extract sub_images based on the training polygons
## make sure ground truth raster already exist
#${eo_dir}/thawslumpScripts/get_sub_images.sh ${para_file}

shape_train=$(python2 ${para_py} -p ${para_file} training_polygons)

rm "sk_svm_trained.pkl" || true
${eo_dir}/planetScripts/planet_svm_classify.py -t -s ${shape_train}

###########pclassification##########
${eo_dir}/planetScripts/planet_svm_classify.py ${input_image}
filename=$(basename "$input_image")
filename_no_ext="${filename%.*}"
#inf_dir=${filename_no_ext}
inf_dir=inf_results

NUM_ITERATIONS=$(python2 ${para_py} -p ${para_file} export_iteration_num)
trail=iter${NUM_ITERATIONS}
expr_name=$(python2 ${para_py} -p ${para_file} expr_name)
testid=$(basename $PWD)_${expr_name}_${trail}
output=${testid}.tif

#output=${filename_no_ext}_classified.tif

cd ${inf_dir}

    #python ${eo_dir}/gdal_class_mosaic.py -o ${output} -init 0 *_pred.tif
    gdal_merge.py -init 0 -n 0 -a_nodata 0 -o ${output} *.tif
    cp ${output} ../.

#    gdal_polygonize.py -8 ${output} -b 1 -f "ESRI Shapefile" ${testid}.shp
#
#    # post processing of shapefile
#    cp ../${para_file}  ${para_file}
#    min_area=$(python2 ${para_py} -p ${para_file} minimum_gully_area)
#    min_p_a_r=$(python2 ${para_py} -p ${para_file} minimum_ratio_perimeter_area)
#    ${deeplabRS}/polygon_post_process.py -p ${para_file} -a ${min_area} -r ${min_p_a_r} ${testid}.shp ${testid}_post.shp

cd ..

# accuracies assessment
${eo_dir}/thawslumpScripts/accuracies_assess.sh ${para_file}


# back up results
mkdir -p result_backup

cp ${para_file} result_backup/${testid}_para.ini
mv otb_acc_log.txt result_backup/${testid}_otb_acc_log.txt
mv ${output} result_backup/${output}
mv planet_svm_log.txt  result_backup/${testid}_planet_svm_log.txt


duration=$SECONDS
echo "$(date): time cost of applying support vector machine : ${duration} seconds">>"time_cost.txt"
