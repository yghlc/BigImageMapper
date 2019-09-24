#!/usr/bin/env python
# Filename: create_inf_image_list 
"""
introduction:  get information list

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 24 September, 2019
"""

import os, sys
HOME = os.path.expanduser('~')
# path of DeeplabforRS
codes_dir2 = HOME +'/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)

import basic_src.io_function as io_function

# qtb_sentinel2_2018_mosaic-0000053760-0000268800_8bit.tif
# qtb_sentinel2_2018_mosaic-0000053760-0000241920_8bit.tif
# qtb_sentinel2_2018_mosaic-0000053760-0000215040_8bit.tif
# qtb_sentinel2_2018_mosaic-0000053760-0000188160_8bit.tif
# qtb_sentinel2_2018_mosaic-0000053760-0000161280_8bit.tif
# qtb_sentinel2_2018_mosaic-0000053760-0000134400_8bit.tif
# qtb_sentinel2_2018_mosaic-0000053760-0000107520_8bit.tif
# qtb_sentinel2_2018_mosaic-0000053760-0000080640_8bit.tif
#
# qtb_sentinel2_2018_mosaic-0000026880-0000268800_8bit.tif
# qtb_sentinel2_2018_mosaic-0000026880-0000241920_8bit.tif
# qtb_sentinel2_2018_mosaic-0000026880-0000215040_8bit.tif
# qtb_sentinel2_2018_mosaic-0000026880-0000188160_8bit.tif
# qtb_sentinel2_2018_mosaic-0000026880-0000161280_8bit.tif
# qtb_sentinel2_2018_mosaic-0000026880-0000134400_8bit.tif
# qtb_sentinel2_2018_mosaic-0000026880-0000107520_8bit.tif
# qtb_sentinel2_2018_mosaic-0000026880-0000080640_8bit.tif

# get image list
image_dir = HOME + '/Data/Qinghai-Tibet/entire_QTP_images/sentinel-2/8bit_dir/sentinel-2_2018_mosaic_v4'

# get image list
image_files = io_function.get_file_list_by_ext('.tif',image_dir,bsub_folder=False)
image_count = len(image_files)


# output all file name
with open('inf_image_list.txt') as list_obj:
    for image_path in image_files:
        list_obj.writelines(os.path.basename(image_path)[1]+'\n')

