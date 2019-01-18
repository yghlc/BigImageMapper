#!/bin/bash

# orthorectify using mapproject (aps)
# run this script in the ZY3 image folder: e.g.,
# ~/Data/Qinghai-Tibet/beiluhe/beiluhe_ZY3/ZY3_NAD_E92.8_N35.0_20141207_L1A0002929919

export PATH=~/programs/StereoPipeline-2.6.1-2018-09-06-x86_64-OSX/bin:$PATH

#dem=~/Data/Qinghai-Tibet/beiluhe/DEM/srtm_30/beiluhe_strm30.tif
dem=~/Data/Qinghai-Tibet/beiluhe/DEM/srtm_30/beiluhe_strm30_prj_utm.tif
tifname=$(ls *-NAD.tiff)
#filename_no_ext
prename="${tifname%.*}"
our_res=2.1

output=${prename}_prj.tif

# ${prename}.xml
# sicne ASP complain the xml is not recognised, then remove it. the script still work without this complaint
mapproject -t rpc $dem ${prename}.tiff  ${output} --mpp ${our_res} \
	--ot UInt16 --tif-compress None --mo 'METADATATYPE=ZY3'


