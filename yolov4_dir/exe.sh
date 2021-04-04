#!/usr/bin/env bash

#introduction: Run the training and prediction of yolov4 using custom dataset
#
#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 2 April, 2021


# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

# any folder defined in yolov4_obj.cfg need to be created first.

# run this in the singularity container
#echo $PATH


# -dont_show flag stops chart from popping up
# -map flag overlays mean average precision on chart to see how accuracy of your model is, only add map flag if you have a validation dataset

#./darknet detector train <path to obj.data> <path to custom config> yolov4.conv.137 -dont_show -map
#darknet detector train obj.data yolov4_obj.cfg yolov4.conv.137 -dont_show -map

# resume training
#darknet detector train obj.data yolov4_obj.cfg train_backup/yolov4_obj_last.weights -dont_show -map

# stop 1-gpu training, then start multiple GPU trianing.
# need to check if get Nan, if yes, need to change learning rates and burn in https://github.com/AlexeyAB/darknet#how-to-train-with-multi-gpu
darknet detector train obj.data yolov4_obj.cfg train_backup/yolov4_obj_1gpu.weights -gpus 0,1,2,3  -dont_show -map



## prediction
#./darknet detector test <path to .data file> <path to config> <path to weights> <path to image>

# #it wil save to "predictions.jpg" or show it if there is display env
#darknet detector test obj.data yolov4_obj.cfg train_backup/yolov4_obj_5000.weights data/test/c7df5232291486a3.jpg
#darknet detector test obj.data yolov4_obj.cfg train_backup/yolov4_obj_5000.weights data/test/53386dae3cf13cc3.jpg -dont_show  -ext_output

## multiple inputs and save results to result.json
#darknet detector test obj.data yolov4_obj.cfg train_backup/yolov4_obj_5000.weights -dont_show  -ext_output -out result.json < test_images.txt
# save result to txt (same things previously on the screen)
#darknet detector test obj.data yolov4_obj.cfg train_backup/yolov4_obj_5000.weights -dont_show  -ext_output < test_images.txt > result.txt

# Checking the Mean Average Precision (mAP) of Your Model
#darknet detector map obj.data yolov4_obj.cfg train_backup/yolov4_obj_5000.weights