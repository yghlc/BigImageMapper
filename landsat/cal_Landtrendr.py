#!/usr/bin/env python
# Filename: cal_Landtrendr.py
"""
introduction: LandTrendr, temporal segmentation algorithms

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 22 June, 2019
"""

import sys,os
from optparse import OptionParser
import rasterio
import numpy as np

# import pandas as pd # read and write excel files

HOME = os.path.expanduser('~')
codes_dir2 = HOME + '/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)

import datetime
from astropy.time import Time
import matplotlib.pyplot as plt

import basic_src.io_function as io_function

from plot_landsat_timeseries import get_date_string_list
from plot_landsat_timeseries import get_msi_file_list

import split_image
from basic_src.RSImage import RSImageclass

import multiprocessing
from multiprocessing import Pool

from scipy import stats

from cal_TheilSen_trend import read_aoi_data

from cal_TheilSen_trend import save_aoi_to_file

from cal_TheilSen_trend import date_string_to_number

from cal_TheilSen_trend import get_month

from cal_TheilSen_trend import filter_time_series_by_month





def main(options, args):

    msi_files = args        # all images in this file should have the same width and height
    # output = options.output
    name_index= options.name_index

    if len(msi_files) < 1:
        raise IOError('NO input images')

    # test_TheilSen()

    # sort the file to make
    msi_files = sorted(msi_files)

    # for file in msi_files:
    #     print(file)

    # test
    aoi = (300, 250, 600, 300)  # (xoff, yoff ,xsize, ysize) in pixels
    # aoi = (300, 250, 10, 20)
    # band_index = [1,2,3]    # for test

    valid_month = [7, 8]
    confidence_inter = 0.95

    # split the image and label
    img_obj = RSImageclass()
    if img_obj.open(msi_files[0]) is False:
        raise IOError('Open %s failed'%msi_files[0])
    width = img_obj.GetWidth()
    height = img_obj.GetHeight()
    patch_w = 200
    patch_h = 200
    patch_boundary = split_image.sliding_window(width, height, patch_w, patch_h, 0, 0)  # boundary of patch (xoff,yoff ,xsize, ysize)


    # use multiple thread
    # num_cores = multiprocessing.cpu_count()
    # print('number of thread %d'%num_cores)
    # # theadPool = mp.Pool(num_cores)  # multi threads, can not utilize all the CPUs? not sure hlc 2018-4-19
    # theadPool = Pool(num_cores)       # multi processes
    #
    # # for idx, aoi in enumerate(patch_boundary):
    # #     print(idx, aoi)
    #
    # tmp_dir = '%s_trend_patches'%name_index
    # parameters_list = [(msi_files, aoi, name_index, valid_month, confidence_inter, os.path.join(tmp_dir,'%d.tif'%idx))
    #                    for idx, aoi in enumerate(patch_boundary)]
    # results = theadPool.map(cal_trend_for_one_index_parallel,parameters_list)


    cal_trend_for_one_index(msi_files, aoi, 'brightness', valid_month, confidence_inter, 'brightness_trend.tif')

    # cal_trend_for_one_index(msi_files, aoi, 'greenness', valid_month, confidence_inter, 'greenness_trend.tif')
    #
    # cal_trend_for_one_index(msi_files, aoi, 'wetness', valid_month, confidence_inter, 'wetness_trend.tif')
    #
    # cal_trend_for_one_index(msi_files, aoi, 'NDVI', valid_month, confidence_inter, 'NDVI_trend.tif')
    #
    # cal_trend_for_one_index(msi_files, aoi, 'NDWI', valid_month, confidence_inter, 'NDWI_trend.tif')
    #
    # cal_trend_for_one_index(msi_files, aoi, 'NDMI', valid_month, confidence_inter, 'NDMI_trend.tif')

    # test = 1

    ### ### ### ### haven\'t complete ### ### ### ### ### ###





if __name__ == "__main__":
    usage = "usage: %prog [options] msi_file1 msi_file2 ..."
    parser = OptionParser(usage=usage, version="1.0 2019-4-14")
    parser.description = 'Introduction: haven\'t complete '

    parser.add_option("-o", "--output",
                      action="store", dest="output",
                      help="the output file path")

    parser.add_option("-n", "--name_index",
                      action="store", dest="name_index",
                      help="the name of mult-spectral index")

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
