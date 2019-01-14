#!/bin/bash

# unpack zy3 images tarball.
# run in ~/Data/Qinghai-Tibet/beiluhe/beiluhe_ZY3

for tar in $(ls *.gz); do

	echo $tar
	filename=$(basename -- "$tar")
	extension="${filename##*.}"
	filename="${filename%.*}"
	
	# rm .tar
	filename="${filename%.*}"	

	echo $filename
	mkdir $filename
	tar -zxvf $tar -C $filename
	
	#exit
done
