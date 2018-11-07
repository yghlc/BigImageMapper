#!/bin/bash

#copy three surface reflectance (SR) images, which can cover most of the area

for order_num in  234347 234346 234345 ; do

	echo ${order_num}
	planet_order=../../orders/planet_order_${order_num}
	#echo ${planet_order}/2018*/*_SR.tif
	ls -l ${planet_order}/2018*/*_SR.tif 
	# copy to current folder
	cp -p ${planet_order}/2018*/*_SR.tif .
done




