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

