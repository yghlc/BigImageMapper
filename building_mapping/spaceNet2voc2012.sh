#!/bin/bash

# convert data format from SpanceNet to VOC format for segmentation.
# run this in /BhaltosMount/Bhaltos/lingcaoHuang/SpaceNet

# script for format conversion.
python_script=${HOME}/codes/PycharmProjects/rsBuildingMapping/SpaceNetChallenge/utilities/python/createDataSpaceNet.py

for spaceNet in spaceNet?; do
  echo $spaceNet
  cd $spaceNet



  cd ..

done


#${spacenet_root}
AOI_2=AOI_2_Vegas_Train
AOI_3=AOI_3_Paris_Train
AOI_4=AOI_4_Shanghai_Train
AOI_5=AOI_5_Khartoum_Train

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


