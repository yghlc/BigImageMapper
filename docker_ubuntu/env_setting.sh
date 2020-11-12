#!/bin/bash

# download the sigularity image
#wget https://www.dropbox.com/s/we9yxv72putq42p/ubuntu16.04_itsc_tf.simg?dl=0 --output-document=ubuntu16.04_itsc_tf.simg

# copy libs
#cp -r /home/hlc/programs/cuda-9.0 ./packages/programs
#cp -r /home/hlc/programs/cuDNN_7.0  ./packages/programs
#mkdir -p ./packages/Data/deeplab/v3+/pre-trained_model
#cp /home/hlc/Data/deeplab/v3+/pre-trained_model/deeplabv3_xception_2018_01_04.tar.gz ./packages/Data/deeplab/v3+/pre-trained_model/.
#mkdir ./packages/bin
#cp /home/hlc/bin/cp_shapefile ./packages/bin/.

# copy this file, then install inside singularity
cp /home/hlc/programs/OTB-6.6.1-Linux64.run .


# copy codes
#git clone https://github.com/yghlc/DeeplabforRS.git ./packages/codes/PycharmProjects/DeeplabforRS
#git clone https://github.com/yghlc/Landuse_DL ./packages/codes/PycharmProjects/Landuse_DL
#git clone https://github.com/yghlc/models.git ./packages/codes/PycharmProjects/tensorflow/yghlc_tf_model 


