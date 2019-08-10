#!/usr/bin/env python
# Filename: replace_images.py 
"""
introduction:

Replace images which have large parts of nodata pixels in sentinel-2_2018_mosaic_v1
using the images (with the same file name) in sentinel-2_2018_mosaic_v2

run in ~/Data/Qinghai-Tibet/entire_QTP_images/sentinel-2

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 10 August, 2019
"""

import os, sys
HOME = os.path.expanduser('~')
# path of DeeplabforRS
codes_dir2 = HOME +'/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)

import basic_src.io_function as io_function

v3_nodata_per = {}
v2_nodata_per = {}

with open('sentinel-2_2018_mosaic_v3/no_data_percentage.txt') as fv3_obj:
    for line in fv3_obj.readlines():
        filename = line.split()[1]
        nodata_per = float(line.split()[1])
        v3_nodata_per[filename] = nodata_per

    print(v3_nodata_per)

with open('sentinel-2_2018_mosaic_v2/no_data_percentage.txt') as fv2_obj:
    for line in fv2_obj.readlines():
        filename = line.split()[1]
        nodata_per = float(line.split()[1])
        v2_nodata_per[filename] = nodata_per

    print(v2_nodata_per)

# get image list
image_files = io_function.get_file_list_by_ext('.tif','sentinel-2_2018_mosaic_v3',bsub_folder=False)
image_count = len(image_files)

files_without_nodata_per = []
files_with_large_diff = []

for idx, img_file in enumerate(image_files):
    img_name = os.path.basename(img_file)

    if img_name in v3_nodata_per.keys():
        diff_per = v3_nodata_per[img_name] - v2_nodata_per[img_name]
        if abs(diff_per) > 10:
            files_with_large_diff.append(img_name)
        else:
            # for the files have large nodata percentage but on the edge of QTP, we should don't need to copy them
            print('%s in v3 and v2 has similar nodata percentage (%2.lf vs %.2lf)'%(img_name,v3_nodata_per[img_name],v2_nodata_per[img_name]))

    else:
        print('%s missed nodata percentage in v3'%img_name)
        files_without_nodata_per.append(img_name)

print("image files with large diffence in nodata percentage:")
for tmp in files_with_large_diff:
    print(tmp)

print("image files missed nodata percentage in v3:")
for tmp in files_without_nodata_per:
    print(tmp)