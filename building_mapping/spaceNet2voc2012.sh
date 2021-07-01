#!/bin/bash

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

# convert training data format from SpanceNet to VOC format for segmentation.
# run this in /BhaltosMount/Bhaltos/lingcaoHuang/SpaceNet

# script for format conversion.
python_script=${HOME}/codes/PycharmProjects/rsBuildingSeg_python3/SpaceNetChallenge/utilities/python/createDataSpaceNet.py
voc_format_dir=/home/lihu9680/Data/temp/SpaceNet_voc

# SpaceNet2
root=spaceNet2
for AOI in $(ls -d ${root}/*train*/*Train*); do
  echo $AOI
  echo training data dir: ${AOI}
  training_data_root=${AOI}
  outputDirectory=${voc_format_dir}/${AOI}
  echo ${training_data_root}
  echo ${outputDirectory}

  mkdir -p ${outputDirectory}
  python ${python_script} ${training_data_root} --convertTo8Bit --trainTestSplit 0.8 \
  --srcImageryDirectory RGB-PanSharpen --outputDirectory ${outputDirectory} --annotationType PASCALVOC2012

#  exit

done



