#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

# run this in the singularity container

echo $PATH

# -dont_show flag stops chart from popping up
# -map flag overlays mean average precision on chart to see how accuracy of your model is, only add map flag if you have a validation dataset

#./darknet detector train <path to obj.data> <path to custom config> yolov4.conv.137 -dont_show -map
darknet detector train obj.data yolov4_obj.cfg yolov4.conv.137 -dont_show -map

