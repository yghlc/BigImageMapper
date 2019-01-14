#!/bin/bash

# unzip spot image upackages
# run in ~/Data/Qinghai-Tibet/beiluhe/beiluhe_spot

for zip in $(ls SWH-*.zip); do 
	
	echo $zip
	filename=$(basename -- "$zip")
	extension="${filename##*.}"
	filename="${filename%.*}"
	echo $filename
	unzip $zip -d $filename	
	#exit
done

# gdalinfo all the images
