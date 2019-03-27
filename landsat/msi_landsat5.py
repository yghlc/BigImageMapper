#!/usr/bin/env python
# Filename: ndvi_landsat5
"""
introduction: calculate multispectral indices of Landsat 5,
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

def cal_ndvi_landsat5(img_path):
    return cal_two_band(img_path, 'B4','B3','NDVI')

def cal_ndwi_landsat5(img_path):
    return cal_two_band(img_path, 'B2','B4','NDWI')

def cal_ndmi_landsat5(img_path):
    return cal_two_band(img_path, 'B4','B5','NDMI')


## coefficents are from paper: Crist, E. P. (1985).
#  A TM tasseled cap equivalent transformation for reflectance factor data.
# Remote Sensing of Environment, 17(3), 301-306.


def cal_brightness_landsat5(img_path):
    brightness_coeff = np.array([0.2043, 0.4158, 0.5524, 0.5741, 0.3124, 0.2303])
    return cal_tasselled_cap(img_path,brightness_coeff,'brightness')

def cal_greenness_landsat5(img_path):
    greenness_coeff = np.array([-0.1603, -0.2819, -0.4934, 0.794, -0.0002, -0.1446])
    return cal_tasselled_cap(img_path, greenness_coeff, 'greenness')

def cal_wetness_landsat5(img_path):
    wetness_coeff = np.array([0.0315, 0.2021, 0.3102, 0.1594, -0.6806, -0.6109])
    return cal_tasselled_cap(img_path, wetness_coeff, 'wetness')



def main(options, args):

    # folder containing images (download from Google Earth Engine)
    # img_folder = args[0]
    img_folder = '/Users/huanglingcao/Data/Qinghai-Tibet/beiluhe/beiluhe_landsat/LT05_2010to2011'
    img_file_list = io_function.get_file_list_by_ext('.tif',img_folder,bsub_folder=False)

    # img_file_list = img_file_list[:2]   # for test
    satellite = 'landsat5'
    # #ndvi
    batch_cal_msi(img_file_list, satellite+'_ndvi.tif', cal_ndvi_landsat5)

    #ndwi
    batch_cal_msi(img_file_list, satellite+'_ndwi.tif', cal_ndwi_landsat5)

    #ndmi
    batch_cal_msi(img_file_list, satellite+'_ndmi.tif', cal_ndmi_landsat5)

    #brightness
    batch_cal_msi(img_file_list, satellite+'_brightness.tif', cal_brightness_landsat5)

    # greenness
    batch_cal_msi(img_file_list, satellite+'_greenness.tif', cal_greenness_landsat5)

    # wetness
    batch_cal_msi(img_file_list, satellite+'_wetness.tif', cal_wetness_landsat5)






    pass

if __name__ == "__main__":
    usage = "usage: %prog [options] image_folder "
    parser = OptionParser(usage=usage, version="1.0 2019-3-26")
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
