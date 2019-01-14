#!/bin/bash

#use gdalinfo to list all the information of spot images.
# run this codes in ~/Data/Qinghai-Tibet/beiluhe/beiluhe_spot

output=imagery_info.txt
for tif in $(ls SWH*/SCEN*/*.TIF); do 
	
	echo $tif
	filename=$(basename -- "$zip")
	extension="${filename##*.}"
	filename="${filename%.*}"
	#echo $filename
	

	echo "file path:" $tif  >> ${output}	
	gdalinfo $tif >>${output}
	# insert a new line
	echo >> ${output}
	echo >> ${output}
done

