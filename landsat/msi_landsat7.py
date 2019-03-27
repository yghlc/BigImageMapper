#!/usr/bin/env python
# Filename: ndvi_landsat7
"""
introduction: calculate multispectral indices of Landsat 7,
including Brightness, Greenness, Wetness, NDVI, NDWI, NDMI

The input image is download from Google Eerth Engine
For comparison, we will stack the NDVI of each image and give a name consiting with image date, pathrow, and 'NDVI'


authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 26 March, 2019
"""

import sys,os
from optparse import OptionParser
import rasterio

import numpy as np

HOME = os.path.expanduser('~')
codes_dir2 = HOME + '/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)

import basic_src.io_function as io_function
import datetime
import struct

# from basic_src.RSImage import RSImageclass

# from msi_landsat8 import get_band_names # get_band_names (img_path):
# from msi_landsat8 import get_band_name  # get_band_name (img_path,pro_name):

from msi_landsat8 import cal_two_band   # cal_two_band (img_path,band1_name,band2_name,pro_name)
from msi_landsat8 import cal_tasselled_cap      # cal_tasselled_cap(img_path,coefficents,pro_name):

# from msi_landsat8 import save_oneband_list  # save_oneband_list(save_path,ndvi_list, band_name_list, org_img):
from msi_landsat8 import batch_cal_msi      # batch_cal_msi(img_file_list, output_name, cal_function):

def cal_ndvi_landsat7(img_path):
    return cal_two_band(img_path, 'B4','B3','NDVI')

def cal_ndwi_landsat7(img_path):
    return cal_two_band(img_path, 'B2','B4','NDWI')

def cal_ndmi_landsat7(img_path):
    return cal_two_band(img_path, 'B4','B5','NDMI')


## coefficents are from paper: Huang, C., Wylie, B., Yang, L., Homer, C., & Zylstra, G. (2002).
#  Derivation of a tasselled cap transformation based on Landsat 7 at-satellite reflectance.
# International Journal of Remote Sensing, 23(8), 1741-1748
# note: the original paper is based on at-satellite reflectance (TOA), but our calculation is based on surface reflectance

def cal_brightness_landsat7(img_path):
    # in Ingmar paper, the third one is 0.3902, but orginal paper is 0.3904
    brightness_coeff = np.array([0.3561, 0.3972, 0.3902, 0.6966, 0.2286, 0.1596])
    return cal_tasselled_cap(img_path,brightness_coeff,'brightness')

def cal_greenness_landsat7(img_path):
    greenness_coeff = np.array([-0.3344, -0.3544, -0.4556, 0.6966, -0.0242, -0.2630])
    return cal_tasselled_cap(img_path, greenness_coeff, 'greenness')

def cal_wetness_landsat7(img_path):
    wetness_coeff = np.array([0.2626, 0.2141, 0.0926, 0.0656, -0.7629, -0.5388])
    return cal_tasselled_cap(img_path, wetness_coeff, 'wetness')



def main(options, args):

    # folder containing images (download from Google Earth Engine)
    # img_folder = args[0]
    img_folder = '/Users/huanglingcao/Data/Qinghai-Tibet/beiluhe/beiluhe_landsat/LE07_2009to2013'
    img_file_list = io_function.get_file_list_by_ext('.tif',img_folder,bsub_folder=False)

    # img_file_list = img_file_list[:2]   # for test
    satellite = 'landsat7'
    # #ndvi
    batch_cal_msi(img_file_list, satellite+'_ndvi.tif', cal_ndvi_landsat7)

    #ndwi
    batch_cal_msi(img_file_list, satellite+'_ndwi.tif', cal_ndwi_landsat7)

    #ndmi
    batch_cal_msi(img_file_list, satellite+'_ndmi.tif', cal_ndmi_landsat7)

    #brightness
    batch_cal_msi(img_file_list, satellite+'_brightness.tif', cal_brightness_landsat7)

    # greenness
    batch_cal_msi(img_file_list, satellite+'_greenness.tif', cal_greenness_landsat7)

    # wetness
    batch_cal_msi(img_file_list, satellite+'_wetness.tif', cal_wetness_landsat7)






    pass

if __name__ == "__main__":
    usage = "usage: %prog [options] image_folder "
    parser = OptionParser(usage=usage, version="1.0 2019-3-23")
    parser.description = 'Introduction: calculate MSI from the image downloaded from Google Earth Engine'

    parser.add_option("-o", "--output",
                      action="store", dest="output",
                      help="the output file path")

    # parser.add_option("-p", "--para",
    #                   action="store", dest="para_file",
    #                   help="the parameters file")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    ## set parameters files
    # if options.para_file is None:
    #     print('error, no parameters file')
    #     parser.print_help()
    #     sys.exit(2)
    # else:
    #     parameters.set_saved_parafile_path(options.para_file)

    main(options, args)
