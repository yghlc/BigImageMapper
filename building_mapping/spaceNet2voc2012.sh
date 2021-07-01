#!/bin/bash

# Exit immediately if a command exits with a non-zero status. E: error trace
set -eE -o functrace

# convert training data format from SpanceNet to VOC format for segmentation.
# run this in /BhaltosMount/Bhaltos/lingcaoHuang/SpaceNet

# script for format conversion.
python_script=${HOME}/codes/PycharmProjects/rsBuildingSeg_python3/SpaceNetChallenge/utilities/python/createDataSpaceNet.py
voc_format_dir=SpaceNet_voc

# SpaceNet2
root=spaceNet2
for AOI in $(ls -d ${root}/*train*/*Train*); do
  echo $AOI
  echo training data dir: ${root}/${AOI}
  training_data_root=${root}/${AOI}
  outputDirectory=${voc_format_dir}/${root}/${AOI}
  echo ${training_data_root}
  echo ${outputDirectory}

  mkdir -p ${outputDirectory}
  python ${python_script} ${training_data_root} --convertTo8Bit --trainTestSplit 0.8 \
  --srcImageryDirectory RGB-PanSharpen --outputDirectory ${outputDirectory} --annotationType PASCALVOC2012

  exit

done


exit



# remove previous success_save.txt file
rm success_save.txt

#echo ${AOIs} ${AOI_3} ${AOI_4} ${AOI_5}
for AOI in ${AOI_2} ${AOI_3} ${AOI_4} ${AOI_5}
do
    echo training data dir: $spacenet_root/$AOI

    #using createDataSpaceNet.py to convert spaceNet file to PASCALVOC2012 format
#    python python/createDataSpaceNet.py /path/to/spacenet_sample/AOI_2_Vegas_Train/ \
#           --srcImageryDirectory RGB-PanSharpen
#           --outputDirectory /path/to/spacenet_sample/annotations/ \
#           --annotationType PASCALVOC2012 \
#           --imgSizePix 400

training_data_root=${spacenet_root}/${AOI}
outputDirectory=${output_root}/${AOI}
echo ${training_data_root}
echo ${outputDirectory}

#using createDataSpaceNet.py to convert spaceNet file to PASCALVOC2012 format
python ${python_script} ${training_data_root} --convertTo8Bit --trainTestSplit 0.8 --srcImageryDirectory RGB-PanSharpen --outputDirectory ${outputDirectory} --annotationType PASCALVOC2012

#using createDataSpaceNet.py to convert spaceNet file to SBD format
#python ${python_script} ${training_data_root} --convertTo8Bit --trainTestSplit 0.8 --srcImageryDirectory RGB-PanSharpen --outputDirectory ${outputDirectory} --annotationType SBD


done


