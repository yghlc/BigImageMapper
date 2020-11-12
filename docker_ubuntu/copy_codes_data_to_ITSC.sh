#!/bin/bash

# copy software, data from Cryo03 to ITSC services
# run this script on Cryo03 Jan 10 2019, hlc

# copy software in programs

# due to the path issue, need to installed again
#scp -r ~/programs/OTB-6.6.1-Linux64 $chpc_host:~/programs/.
scp -r ~/programs/OTB-6.6.1-Linux64.run $chpc_host:~/programs/.

# need to tar then, then unpcakge. scp will result in more storage, don't know why
scp -r ~/programs/cuda-9.0 $chpc_host:~/programs/.
scp -r ~/programs/cuDNN_7.0 $chpc_host:~/programs/.
# need to change "/home/hlc" in ~/programs/anaconda2/bin/*.py,
# using sed, e.g. sed -i -e  s%/home/hlc%/users/s1155090023%g gdal_edit.py
scp -r ~/programs/anaconda2 $chpc_host:~/programs/.


# copy pre-trained model of deeplab
# on chpc: mkdir -p ~/Data/deeplab/v3+/pre-trained_model
scp ~/Data/deeplab/v3+/pre-trained_model/deeplabv3_xception_2018_01_04.tar.gz $chpc_host:~/Data/deeplab/v3+/pre-trained_model/.

# copy planet data
# on chpc: mkdir -p ~/Data/Qinghai-Tibet/beiluhe/beiluhe_planet/beiluhe_basin/201805
scp 20180522_035755_3B_AnalyticMS_SR_mosaic_8bit_rgb_basinExt.tif $chpc_host:~/Data/Qinghai-Tibet/beiluhe/beiluhe_planet/beiluhe_basin/201805/


# copy shapefile and DEM
scp -r DEM thaw_slumps $chpc_host:~/Data/Qinghai-Tibet/beiluhe/.

# singularity container
cd ~/codes/PycharmProjects/Landuse_DL/docker_ubuntu1604
scp ubuntu16.04_itsc_tf.simg $chpc_host:~/codes/PycharmProjects/Landuse_DL/docker_ubuntu1604/.

#copy files in
dst=/users/s1155090023/Data/Qinghai-Tibet/beiluhe/beiluhe_planet/autoMapping/BLH_basin_deeplabV3+_1
cd ~/Data/Qinghai-Tibet/beiluhe/beiluhe_planet/autoMapping/BLH_basin_deeplabV3+_1
scp exe.sh $chpc_host:$dst
scp run_INsingularity.sh $chpc_host:$dst
# need to change "/home/hlc" in the para.ini
scp para.ini $chpc_host:$dst
scp inf_image_list.txt $chpc_host:$dst

scp ~/bin/cp_shapefile $chpc_host:bin/cp_shapefile






