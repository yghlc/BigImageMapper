#!/usr/bin/env python
# Filename: crop_images_to_valid_extent.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 05 February, 2022
"""

import os,sys

code_dir = os.path.expanduser('~/codes/PycharmProjects/DeeplabforRS')
sys.path.insert(0, code_dir)

import basic_src.io_function as io_function

from flooding_proc_accuracy import resample_crop_raster_using_shp


def crop_houston_compose_3bands_VH_VV_VHVV_RGB():
    rgb_tif_dir = os.path.expanduser('~/Data/flooding_area/Houston/compose_3bands_VH_VV_VHVV')
    image1_path = os.path.join(rgb_tif_dir,'20170829_RGB_composite_prj_8bit.tif')
    image2_path = os.path.join(rgb_tif_dir,'20170829_RGB_composite_north_prj_8bit.tif')

    valid_shp = os.path.expanduser('~/Data/flooding_area/Houston/extent_image_valid/houston_valid_image_extent.shp')

    img1_output = io_function.get_name_by_adding_tail(image1_path,'crop')
    img2_output = io_function.get_name_by_adding_tail(image2_path,'crop')

    resample_crop_raster_using_shp(valid_shp, image1_path, img1_output,dst_nondata=0)
    resample_crop_raster_using_shp(valid_shp, image2_path, img2_output,dst_nondata=0)

def main():
    crop_houston_compose_3bands_VH_VV_VHVV_RGB()

if __name__ == '__main__':
    main()