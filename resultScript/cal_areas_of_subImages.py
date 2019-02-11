#!/usr/bin/env python
# Filename: convert_xy_inQGIS_composer.py
"""
introduction: calculate the area of subImages (with different class)



authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 11 February, 2019
"""

import os, sys
HOME = os.path.expanduser('~')
# path of DeeplabforRS
codes_dir2 =  HOME +'/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)

import basic_src.io_function as io_function
import basic_src.basic as basic
import basic_src.RSImage as RSImage
from basic_src.RSImage import RSImageclass

# folder='/home/hlc/Data/Qinghai-Tibet/beiluhe/beiluhe_planet/autoMapping/BLH_basin_deeplabV3+_1/subImages'
folder='/Users/huanglingcao/Dropbox/a_canada_sync/subImages'

thawslump_subImages = io_function.get_file_list_by_pattern(folder,"*_class_1.tif")
nonthawslump_subImages = io_function.get_file_list_by_pattern(folder,"*_class_0.tif")

# print(thawslump_subImages)
# print(nonthawslump_subImages)

def get_area(img_path):

    valid_pixel_count = RSImage.get_valid_pixel_count(img_path)

    # get resolution
    img_obj = RSImageclass()
    if img_obj.open(img_path) is False:
        return False
    res_x = img_obj.GetXresolution()
    res_y = img_obj.GetYresolution()

    area = abs(res_x*res_y*valid_pixel_count)/pow(10,6)
    return area

def mosaic_images(img_list, output):
    if os.path.isfile(output) is True:
        print('warning, %s already exist, skip'%output)
        return True

    args_list=['gdal_merge.py','-init','0','-a_nodata','0','-o',output]
    args_list.extend(img_list)

    if basic.exec_command_args_list_one_file(args_list,output) is False:
        raise IOError('output not exist')
    return True


output_1='class_1.tif'
mosaic_images(thawslump_subImages,output_1)
area_class_1 = get_area(output_1)


output_0='class_0.tif'
mosaic_images(nonthawslump_subImages,output_0)
area_class_0 = get_area(output_0)


entire_img=HOME+'/Data/Qinghai-Tibet/beiluhe/beiluhe_planet/beiluhe_basin/201805/20180522_035755_3B_AnalyticMS_SR_mosaic_8bit_rgb_basinExt.tif'

area_entire_area = get_area(entire_img)

print("area of class 1: %.4f, class 0: %.4f, and the entire area: %.4f"%(area_class_1,area_class_0,area_entire_area))

print("Percent of class 1: %.4f, class 0: %.4f, and total: %.4f"%
      (100*area_class_1/area_entire_area,100*area_class_0/area_entire_area,100*(area_class_1+area_class_0)/area_entire_area))

# info for the entire image
# Size is 30916, 18713
# Pixel Size = (3.000000000000000,-3.000000000000000
print(30916*18713*3.00*3.00/pow(10,6))