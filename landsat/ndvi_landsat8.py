#!/usr/bin/env python
# Filename: ndvi_landsat8 
"""
introduction: calculate NDVI from landsat 8
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


def cal_ndvi_landsat8(img_path):

    band_names = get_band_names(img_path)
    with rasterio.open(img_path) as src:
        band_nir = src.read(band_names.index('B5')+1)
        band_red = src.read(band_names.index('B4')+1)

        # Allow division by zero
        np.seterr(divide='ignore', invalid='ignore')
        # Calculate NDVI
        ndvi = (band_nir.astype(float) - band_red.astype(float)) / (band_nir + band_red)
        nan_loc = np.where(np.logical_or(ndvi < -1, ndvi > 1))
        ndvi[nan_loc] = np.nan
        ndvi = ndvi.astype(np.float32)  # to float 32
        out_band_name = get_band_name(img_path, 'NDVI')

        return ndvi, out_band_name

def save_ndvi_list(save_path,ndvi_list, band_name_list, org_img):
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



def main(options, args):

    # folder containing images (download from Google Earth Engine)
    img_folder = args[0]
    img_file_list = io_function.get_file_list_by_ext('.tif',img_folder,bsub_folder=False)

    # img_file_list = img_file_list[:2]   # for test

    # calculate ndvi
    out_bandname_list = []
    ndvi_list = []
    for img_file in img_file_list:
        ndvi, bandname =  cal_ndvi_landsat8(img_file)
        out_bandname_list.append(bandname)
        ndvi_list.append(ndvi)

    # save to file
    output = options.output
    save_ndvi_list(output,ndvi_list,out_bandname_list,img_file_list[0])



    pass

if __name__ == "__main__":
    usage = "usage: %prog [options] image_folder "
    parser = OptionParser(usage=usage, version="1.0 2019-3-23")
    parser.description = 'Introduction: calculate NDVI from the image downloaded from Google Earth Engine'

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
