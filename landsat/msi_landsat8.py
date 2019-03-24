#!/usr/bin/env python
# Filename: ndvi_landsat8 
"""
introduction: calculate multispectral indices of Landsat 8,
including Brightness, Greenness, Wetness, NDVI, NDWI, NDMI

The input image is download from Google Eerth Engine
For comparison, we will stack the NDVI of each image and give a name consiting with image date, pathrow, and 'NDVI'


authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 23 March, 2019
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

from basic_src.RSImage import RSImageclass


def get_band_names(img_path):
    '''
    get the all the band names (description) in this raster
    :param img_path:
    :return:
    '''
    rs_obj = RSImageclass()
    rs_obj.open(img_path)
    names = rs_obj.Getband_names()
    return names

    ##RasterBand.SetDescription(BandName) # This sets the band name!

def get_band_name(img_path,pro_name):
    file_name = os.path.splitext(os.path.basename(img_path))[0]
    tmp = file_name.split('_')

    path_row = tmp[1]
    date_str = tmp[2]
    date_time_obj = datetime.datetime.strptime(date_str, '%Y%m%d')
    # print(date_time_obj)
    return date_time_obj.strftime('%Y-%m-%d')+ '_' + str(path_row) + '_' + pro_name

def cal_two_band(img_path,band1_name,band2_name,pro_name):
    band_names = get_band_names(img_path)
    with rasterio.open(img_path) as src:
        band_nir = src.read(band_names.index(band1_name) + 1)
        band_red = src.read(band_names.index(band2_name) + 1)
        # Allow division by zero
        np.seterr(divide='ignore', invalid='ignore')
        # Calculate NDVI
        ndvi = (band_nir.astype(float) - band_red.astype(float)) / (band_nir + band_red)
        nan_loc = np.where(np.logical_or(ndvi < -1, ndvi > 1))
        ndvi[nan_loc] = np.nan
        ndvi = ndvi.astype(np.float32)  # to float 32
        out_band_name = get_band_name(img_path, pro_name)

        return ndvi, out_band_name

def cal_ndvi_landsat8(img_path):
    return cal_two_band(img_path, 'B5','B4','NDVI')

def cal_ndwi_landsat8(img_path):
    return cal_two_band(img_path, 'B3','B5','NDWI')

def cal_ndmi_landsat8(img_path):
    return cal_two_band(img_path, 'B5','B6','NDMI')


def cal_tasselled_cap(img_path,coefficents,pro_name):
    # should have six bands: Blue, Green, Red, NIR, SWIR-1, SWIR-2
    band_names = get_band_names(img_path)
    if len(band_names) != 6:
        raise ValueError('require 6 bands in order: Blue, Green, Red, NIR, SWIR-1, SWIR-2')
    with rasterio.open(img_path) as src:

        data_list = [src.read(band_names.index(name) + 1) for name in band_names]
        spectral_data = np.stack(data_list, axis=2)  #stack

        np.seterr(divide='ignore', invalid='ignore')

        # If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.
        tc_values = np.dot(spectral_data, coefficents)
        tc_values = tc_values/10000     # 10000 is the scale of surface reflectance
        tc_values = tc_values.astype(np.float32)

        out_band_name = get_band_name(img_path, pro_name)

        return tc_values, out_band_name


## coefficents are from paper: Baig, M. H. A., Zhang, L., Shuai, T., & Tong, Q. (2014).
#  Derivation of a tasselled cap transformation based on Landsat 8 at-satellite reflectance.
# Remote Sensing Letters, 5(5), 423-431.

def cal_brightness_landsat8(img_path):
    brightness_coeff = np.array([0.3029, 0.2786, 0.4733, 0.5599, 0.508, 0.1872])
    return cal_tasselled_cap(img_path,brightness_coeff,'brightness')


def cal_greenness_landsat8(img_path):
    greenness_coeff = np.array([-0.2941, -0.243, -0.5424, 0.7276, 0.0713, -0.1608])
    return cal_tasselled_cap(img_path, greenness_coeff, 'greenness')

def cal_wetness_landsat8(img_path):
    wetness_coeff = np.array([0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559])
    return cal_tasselled_cap(img_path, wetness_coeff, 'wetness')


def save_oneband_list(save_path,ndvi_list, band_name_list, org_img):
    '''
    save ndvi of multiple images to a file
    :param save_path:
    :param ndvi_list:
    :param band_name_list:
    :param org_img:
    :return:
    '''
    if len(ndvi_list) < 1 or len(band_name_list) < 1:
        raise ValueError('the input of ndvi or name list is empty')

    if len(ndvi_list) != len(band_name_list):
        raise ValueError('the length of ndvi list and band name list is different')

    rsImg_org = RSImageclass()
    rsImg_org.open(org_img)
    org_width = rsImg_org.GetWidth()
    org_height = rsImg_org.GetHeight()
    datetype = rsImg_org.GetGDALDataType()
    prj = rsImg_org.GetProjection()
    geo_tran = rsImg_org.GetGeoTransform()

    save_bandcount = len(ndvi_list)


    rsImg_obj = RSImageclass()
    if rsImg_obj.New(save_path,org_width,org_height,save_bandcount, 6):  # 6 for float 32
        for idx in range(save_bandcount):
            height, width = ndvi_list[idx].shape

            ndvi_str = ndvi_list[idx].tobytes()# struct.pack('%f' % width * height, *templist)
            if rsImg_obj.WritebandData(idx + 1,0,0,width,height,ndvi_str,6):
                rsImg_obj.set_band_name(idx +1 ,band_name_list[idx])

        # set projection and transform
        rsImg_obj.SetProjection(prj)
        rsImg_obj.SetGeoTransform(geo_tran)

def batch_cal_msi(img_file_list, output_name, cal_function):
    # calculate ndvi
    out_bandname_list = []
    ndvi_list = []
    for img_file in img_file_list:
        # ndvi, bandname = cal_ndvi_landsat8(img_file)
        ndvi, bandname = cal_function(img_file)
        out_bandname_list.append(bandname)
        ndvi_list.append(ndvi)
    # save to file
    output = output_name #options.output
    save_oneband_list(output, ndvi_list, out_bandname_list, img_file_list[0])

def main(options, args):

    # folder containing images (download from Google Earth Engine)
    img_folder = args[0]
    img_file_list = io_function.get_file_list_by_ext('.tif',img_folder,bsub_folder=False)

    # img_file_list = img_file_list[:2]   # for test

    # ndvi
    # batch_cal_msi(img_file_list, 'landsat8_ndvi.tif', cal_ndvi_landsat8)

    # ndwi
    # batch_cal_msi(img_file_list, 'landsat8_ndwi.tif', cal_ndwi_landsat8)

    # ndmi
    # batch_cal_msi(img_file_list, 'landsat8_ndmi.tif', cal_ndmi_landsat8)

    # brightness
    # batch_cal_msi(img_file_list, 'landsat8_brightness.tif', cal_brightness_landsat8)

    # greenness
    batch_cal_msi(img_file_list, 'landsat8_greenness.tif', cal_greenness_landsat8)

    # wetness
    batch_cal_msi(img_file_list, 'landsat8_wetness.tif', cal_wetness_landsat8)






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
