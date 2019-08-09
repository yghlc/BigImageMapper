#!/usr/bin/env python
# Filename: non_data_percentage 
"""
introduction: get the percentage of non-data in each images

# run this script in ~/Data/Qinghai-Tibet/entire_QTP_images/sentinel-2/8bit_dir/sentinel-2_2018_mosaic_v3

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 09 August, 2019
"""

import os,sys

HOME = os.path.expanduser('~')
# path of DeeplabforRS
codes_dir2 = HOME +'/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)


import basic_src.RSImage as RSImage
from basic_src.RSImage import RSImageclass
import basic_src.io_function as io_function
import basic_src.basic as basic

# basic.setlogfile('no_data_percentage.txt')

nodata = 0

# get image list
image_files = io_function.get_file_list_by_ext('.tif','.',bsub_folder=False)
image_count = len(image_files)

# save to file
f_obj = open('no_data_percentage.txt','w')

for idx, img_file in enumerate(image_files):
    print('start working on (%d / %d) images'%(idx+1, image_count))
    # band_num  = 1   # only consider the first band
    # bucket_count, hist_min, hist_max, hist_buckets = RSImage.get_image_histogram_oneband(img_file, band_num)

    try:
        valid_count = RSImage.get_valid_pixel_count(img_file)       # count the pixels without nodata pixel
    except IOError:
        basic.outputlogMessage("get grey histrogram of %s failed"%image_files)

    # get image width and height
    img_obj = RSImageclass()
    if img_obj.open(img_file):
        width = img_obj.GetWidth()
        height = img_obj.GetHeight()
        nodata_per = 100.0*(width*height - valid_count)/(width*height)
        basic.outputlogMessage('Nodata percentage %.2lf'%nodata_per)

        img_name = os.path.basename(img_file)
        f_obj.writelines("%d: %s Nodata percentage: %.2lf \n"%(idx+1, img_name, nodata_per))
        f_obj.flush()


f_obj.close()









