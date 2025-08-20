#!/usr/bin/env bash

#introduction: Run the whole process of identifying landforms from remote sensing images using yolov4
#
#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 20 August, 2025

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace


# set GPUs want to used (e.g. use GPU 0 & 1)
#export CUDA_VISIBLE_DEVICES=0,1

# the main parameter files
para_file=main_para.ini

# BigImageMapper scripts dir
eo_dir=~/codes/PycharmProjects/BigImageMapper

################################################
#extract sub_images based on the training polygons
rm time_cost.txt || true


################################################
# preparing training data.
rm -r training_data
${eo_dir}/obj_detection/obtain_objDet_training_data.py ${para_file}

#${eo_dir}/yolov4_dir/pre_yolo_data.py ${para_file}


################################################
## run within ultralytics environment
${eo_dir}/yolov8_dir/pre_yolov8_yaml.py ${para_file}
exit
## training 
#yolo cfg=yolo_conf.yaml
#yolo train model=yolo11n.pt data=yolov8_data.yaml epochs=100 imgsz=640


#exit
### prediction
#rm -r multi_inf_results || true
${eo_dir}/yolov8_dir/predict_yolov8.py ${para_file}
################################################


## post processing and copy results, inf_post_note indicate notes for inference and post-processing
inf_post_note=1
${eo_dir}/yolov4_dir/postProc_yolo.py ${para_file}  ${inf_post_note}
