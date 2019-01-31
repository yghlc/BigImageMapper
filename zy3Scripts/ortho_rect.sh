#!/bin/bash

# orthorectify using mapproject (aps)
# run this script in the ZY3 image folder contaning NAD images: e.g.,
# ~/Data/Qinghai-Tibet/beiluhe/beiluhe_ZY3/ZY3_NAD_E92.8_N35.0_20141207_L1A0002929919

#export PATH=~/programs/StereoPipeline-2.6.1-2018-09-06-x86_64-OSX/bin:$PATH
export PATH=~/programs/StereoPipeline-2.6.1-2018-09-06-x86_64-Linux/bin:$PATH

#number of thread of to use, 8 or 16 on linux, 4 on mac (default is 4)
num_thr=8

#dem=~/Data/Qinghai-Tibet/beiluhe/DEM/srtm_30/beiluhe_strm30.tif
dem=~/Data/Qinghai-Tibet/beiluhe/DEM/srtm_30/beiluhe_strm30_prj_utm.tif
tifname=$(ls *-NAD.tiff)
#filename_no_ext
prename="${tifname%.*}"
our_res=2.1

str=$(basename $PWD)
IFS='_ ' read -r -a array <<< "$str"
output=${array[0]}_${array[4]}_${array[5]}_prj.tif

#output=${prename}_prj.tif

# ${prename}.xml
# sicne ASP complain the xml is not recognised, then remove it. the script still work without this complaint
# on Linux, no METADATATYPE, we set METADATATYPE=ZY3,
# on Mac, it set METADATATYPE=ZY3, and can not be modified, and mapproject copy the rpb files
# Same version, but different behaviour in Mac and Linux, strange
mapproject -t rpc $dem ${prename}.tiff  ${output} --mpp ${our_res} \
	--ot UInt16 --tif-compress None --mo METADATATYPE=ZY3 --threads ${num_thr}


