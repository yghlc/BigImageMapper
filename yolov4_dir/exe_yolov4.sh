#!/usr/bin/env bash

#introduction: Run the whole process of identifying landforms from remote sensing images using yolov4
#
#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 4 April, 2021

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace


# set GPUs want to used (e.g. use GPU 0 & 1)
export CUDA_VISIBLE_DEVICES=0,1

# the main parameter files
para_file=main_para.ini

# Landuse_DL scripts dir
eo_dir=~/codes/PycharmProjects/Landuse_DL

################################################
SECONDS=0
# remove previous data or results if necessary
rm time_cost.txt || true
${eo_dir}/workflow/remove_previous_data.py ${para_file}

#extract sub_images based on the training polygons
${eo_dir}/workflow/get_sub_images_multi_regions.py ${para_file}

################################################
## preparing training data.
${eo_dir}/workflow/split_sub_images.py ${para_file}

${eo_dir}/workflow/split_train_val.py ${para_file}
${eo_dir}/yolov4_dir/pre_yolo_data.py ${para_file}

#${eo_dir}/workflow/training_img_augment.py ${para_file}

duration=$SECONDS
echo "$(date): time cost of preparing training data: ${duration} seconds">>"time_cost.txt"

################################################
## training
# -dont_show flag stops chart from popping up
# -map flag overlays mean average precision on chart to see how accuracy of your model is, only add map flag if you have a validation dataset
darknet detector train data/obj.data yolov4_obj.cfg yolov4.conv.137 -dont_show -map


################################################
#compare model


################################################
### prediction
#rm -r multi_inf_results || true
#${eo_dir}/workflow/parallel_prediction.py ${para_file}


## post processing and copy results, inf_post_note indicate notes for inference and post-processing
#inf_post_note=1
#${eo_dir}/workflow/postProcess.py ${para_file}  ${inf_post_note}


