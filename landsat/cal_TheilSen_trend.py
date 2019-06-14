#!/usr/bin/env python
# Filename: cal_TheilSen_trend
"""
introduction: Apply Theil-Sen Regression to landsat time series

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 12 June, 2019
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
import basic_src.RSImage as RSImage
from msi_landsat8 import get_band_names  # get_band_names(img_path)

import pandas as pd

from plot_landsat_timeseries import get_date_string_list
from plot_landsat_timeseries import get_msi_file_list
from plot_landsat_timeseries import read_time_series

from scipy import stats

def test_TheilSen():

    # example from:
    # https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.mstats.theilslopes.html

    x = np.linspace(-5, 5, num=150)
    y = x + np.random.normal(size=x.size)
    y[11:15] += 10  # add outliers
    y[-5:] -= 7

    #Compute the slope, intercept and 90% confidence interval. For comparison, also compute the least-squares fit with linregress:
    lsq_res = stats.linregress(x, y)

    con_b, ts_slope, lower_slope, upper_slope = TheilSen_regression(x,y, 0.9)

    # res = stats.theilslopes(y, x, 0.90)
    # print(res)
    # res = stats.theilslopes(y, x, 0.10) # change the confidence interval to 10%, the output is the same as 90%
    # print(res)
    # # print(lsq_res)

    # Plot the results.
    # The Theil-Sen regression line is shown in red,
    # with the dashed red lines illustrating the confidence interval of the slope
    # (note that the dashed red lines are not the confidence interval of the regression as
    # the confidence interval of the intercept is not included). The green line shows the least-squares fit for comparison.

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, 'b.')
    # ax.plot(x, res[1] + res[0] * x, 'r-')
    # ax.plot(x, res[1] + res[2] * x, 'r--')
    # ax.plot(x, res[1] + res[3] * x, 'r--')

    ax.plot(x, con_b + ts_slope * x, 'r-')
    ax.plot(x, con_b + lower_slope * x, 'r--')
    ax.plot(x, con_b + upper_slope * x, 'r--')

    ax.plot(x, lsq_res[1] + lsq_res[0] * x, 'g-')


    # plt.show()
    output='test_TheilSen.png'
    plt.savefig(output,bbox_inches="tight") # dpi=200, ,dpi=300


def TheilSen_regression(x,y,confidence_inter):
    '''
    apply Theil-Sen estimator to points
    :param x: x values of points
    :param y: y values of points
    :param confidence_inter: the conficence interval, notes: 0.1 and 0.9 have the same output
    :return: # constant b, slope, lower slope, upper slope
    '''


    res = stats.theilslopes(y, x, confidence_inter)

    return res[1], res[0], res[2], res[3] # constant b, slope, lower slope, upper slope

def read_aoi_data(product_list,date_str_list, aoi_win,bands=None):
    '''
    read a time series from sub-images
    :param file_list: the file list containing brightness
    :param date_str_list: the date string
    :param aoi_win: the windows  #(xoff,yoff ,xsize, ysize) in pixels
    :param bands: the band indeces, default is all of them, bands for test purpose
    :return: two list containing containg date string and numpy arrays
    '''

    if len(product_list) != len(date_str_list):
        raise ValueError('product files and date string not match')

    window = ((aoi_win[1],aoi_win[1]+aoi_win[3]), (aoi_win[0],aoi_win[0]+aoi_win[2]))

    aoi_date_str_list = []
    aoi_series_value = []

    for img_file, date_string in zip(product_list, date_str_list):

        with rasterio.open(img_file) as img_obj:
            # read the all bands
            if bands is None:
                indexes = img_obj.indexes
            else:
                indexes = bands
            data = img_obj.read(indexes, window=window)

            # replace the nodata as zeros (means background)
            # if img_obj.profile.has_key('nodata'):
            # if 'nodata' in img_obj.profile.keys():  # python3 & 2, has_key was removed in python3.x
            #     nodata = img_obj.profile['nodata']
            #     data[np.where(data == nodata)] = 0

            #filter the date by band indices
            sub_date_str_list = date_string
            if bands is not None:
                sub_date_str_list = [ date_str_list[idx - 1] for idx in indexes ]

            aoi_series_value.append(data)
            aoi_date_str_list.append(sub_date_str_list)

    return aoi_date_str_list, aoi_series_value

def save_aoi_to_file(ori_file, aoi_win, arrays_list, save_path_list):
    '''
    save the data of an aoi to file
    :param ori_file: the original file providing metedata , where the subset come from
    :param aoi_win: the windows  #(xoff,yoff ,xsize, ysize) in pixels
    :param arrays_list: array list
    :param save_path: save path
    :return: True if successful, false otherwise
    '''

    # get profile, then update it
    window = ((aoi_win[1], aoi_win[1] + aoi_win[3]), (aoi_win[0], aoi_win[0] + aoi_win[2]))
    with rasterio.open(ori_file) as org:
        profile = org.profile
        new_transform = org.window_transform(window)
    # calculate new transform and update profile (transform, width, height)

    xsize = aoi_win[2]
    ysize = aoi_win[3]

    # save all the arrays in the list
    for img_data,save_path in zip(arrays_list,save_path_list):

        # check width and height
        if img_data.ndim == 2:
            img_data = img_data.reshape(1, img_data.shape)
        nband, height, width = img_data.shape
        if xsize != width or ysize != height:
            raise ValueError("Error, the Size of the saved numpy array is different from the original patch")


        # update profile
        profile.update(count=nband, transform=new_transform, width=xsize, height=ysize)
        # set the block size, it should be a multiple of 16 (TileLength must be a multiple of 16)
        # if profile.has_key('blockxsize') and profile['blockxsize'] > xsize:
        if 'blockxsize' in profile and profile['blockxsize'] > xsize: # python3 & 2
            if xsize % 16 == 0:
                profile.update(blockxsize=xsize)
            else:
                profile.update(blockxsize=16)  # profile.update(blockxsize=16)
        if 'blockysize' in profile and profile['blockysize'] > ysize:
            if ysize % 16 == 0:
                profile.update(blockysize=ysize)
            else:
                profile.update(blockysize=16)  # profile.update(blockxsize=16)

        #
        with rasterio.open(save_path, "w", **profile) as dst:
            # dst.write(img_data, 1)
            for idx, band in enumerate(img_data):  #shape: nband,height,width
                # apply mask
                # band = band*could_mask
                # save
                dst.write_band(idx+1, band)

    return True

def date_string_to_number(date_string,format='%Y-%m-%d'):
    '''
    get the Modified Julian Day (MJD)
    :param date_string:
    :param format:
    :return:
    '''

    date_obj = datetime.datetime.strptime(date_string, format)
    t = Time(date_obj)
    # print(t)
    # print(t.jd)     # JULIAN DAY
    # print(t.mjd)    # Modified Julian Day  MJD
    return int(t.mjd)

def get_month(date_string,format='%Y-%m-%d'):
    date_obj = datetime.datetime.strptime(date_string, format)
    return date_obj.month

def filter_time_series_by_month(date_string_list, arrays_list, keep_month):
    '''
    filter the data only keep particluar months
    :param date_string_list: 2D list
    :param arrays_list: a list contian 3D arrays
    :param keep_month: keep month, e.g, [7,8]
    :return: the lists after filtering
    '''

    out_date_string_list = []
    out_arrays_list = []

    for date_strings, arrays in zip(date_string_list,arrays_list):

        removed_idx = []
        keep_date_strings = []
        for idx, date_str in enumerate(date_strings):
            if get_month(date_str) not in keep_month:
                removed_idx.append(idx)
            else:
                keep_date_strings.append(date_str)

        out_date_string_list.append(keep_date_strings)

        # remove the bands in arrays
        keep_arrays = np.delete(arrays,removed_idx,axis=0)
        out_arrays_list.append(keep_arrays)

    return out_date_string_list, out_arrays_list


def cal_Theilsen_trend(date_string_list,arrays_list,confidence_inter=0.9):
    '''
    calculate Theil sen trend
    :param date_string_list: 2d list, contains date string
    :param arrays_list: 1d list, contains arrays
    :param confidence_inter: the conficence interval, notes: 0.1 and 0.9 have the same output
    :return: a numpy array of trend
    '''

    # convert to x (date), y (observation) array
    # for date_strs in date_string_list:
    #     for item in date_strs:
    #         # print(item)
    #         print(date_string_to_number(item))

    date_num = [ date_string_to_number(item) for date_strs in date_string_list for item in date_strs ]       # x

    # for num in date_num:
    #     print(num)

    # get y
    # obser_value = np.stack(arrays_list,axis=0)  #Join a sequence of arrays along a new axis.
    obser_value = np.concatenate(arrays_list, axis=0)   #Join a sequence of arrays along an existing axis.

    ncount, height, width = obser_value.shape

    # calcuate trend
    output_trend = np.zeros((4,height, width)) # slope, lower slope, upper slope, and intercept
    for row in range(height):
        for col in range(width):

            x = np.array(date_num) #date_num.copy()
            y = obser_value[:,row, col]

            # remove nan value
            not_nan_loc = np.logical_not(np.isnan(y))
            x = x[not_nan_loc]
            y = y[not_nan_loc]


            # perform calculation
            constant, slope, lower_slope, upper_slope = TheilSen_regression(x,y,confidence_inter)

            # test on np median: difference between np.median and the index-based median.
            # because when numpy will average the terms in the middle if total no. of terms are even
            y_median_idx = np.argsort(y)[len(y) // 2]
            # print('median x',np.median(x),'median y',np.median(y),'median x, y',x[y_median_idx],y[y_median_idx])

            intercept = y[y_median_idx] -  slope*x[y_median_idx]

            output_trend[0, row, col] = slope
            output_trend[1, row, col] = lower_slope
            output_trend[2, row, col] = upper_slope
            output_trend[3, row, col] = intercept

            # test = 1


    return output_trend.astype(np.float32)

def cal_trend_for_one_index(msi_files, aoi,index_name,keep_month,confidence,output):
    '''
    calculate the trend of one index
    :param msi_files: multi spectural indces
    :param aoi: the aoi window for calculation # (xoff, yoff ,xsize, ysize) in pixels
    :param index_name: e.g., brightness
    :param keep_month: the months for filtering index values, e.g., [7,8]
    :param confidence: confidence interval for TheilSen regression
    :param output: save path
    :return:
    '''
    # read brightness values
    brightness_files = get_msi_file_list(msi_files, index_name) #'brightness'

    if len(brightness_files) < 1:
        raise ValueError('input files do not contain %s'%index_name)

    # brightness_files = brightness_files[1:2] # for test
    brightness_date_str_list = get_date_string_list(brightness_files)
    # b_date_str, b_value = read_time_series(brightness_files,brightness_date_str_list,x,y, xy_srs)

    # for item in brightness_date_str_list:
    #     for sub_item in item:
    #         print(sub_item)

    # date_int = [date_string_to_number(date_str) for date_str in brightness_date_str_list[0]]
    # print(brightness_date_str_list[0])
    # print(date_int)

    band_index = None
    date_string_list, arrays_list = read_aoi_data(brightness_files, brightness_date_str_list, aoi, bands=band_index)

    # save the trend
    # print(date_string_list)
    # print(arrays_list)
    # print(len(date_string_list))
    # print(len(arrays_list))
    # save_files = ['save_%d.tif'%idx for idx in range(len(brightness_files))]
    # save_aoi_to_file(brightness_files[0],aoi,arrays_list,save_files)

    # filter the months, only keep month 7 and 8.
    date_string_list, arrays_list = filter_time_series_by_month(date_string_list, arrays_list, keep_month)

    # calculate the trend
    trend = cal_Theilsen_trend(date_string_list, arrays_list, confidence_inter=confidence)

    save_files = [output]
    save_aoi_to_file(brightness_files[0], aoi, [trend], save_files)

    return True

def main(options, args):

    msi_files = args

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

    cal_trend_for_one_index(msi_files, aoi, 'brightness', valid_month, confidence_inter, 'brightness_trend.tif')

    cal_trend_for_one_index(msi_files, aoi, 'greenness', valid_month, confidence_inter, 'greenness_trend.tif')

    cal_trend_for_one_index(msi_files, aoi, 'wetness', valid_month, confidence_inter, 'wetness_trend.tif')

    cal_trend_for_one_index(msi_files, aoi, 'NDVI', valid_month, confidence_inter, 'NDVI_trend.tif')

    cal_trend_for_one_index(msi_files, aoi, 'NDWI', valid_month, confidence_inter, 'NDWI_trend.tif')

    cal_trend_for_one_index(msi_files, aoi, 'NDMI', valid_month, confidence_inter, 'NDMI_trend.tif')

    test = 1





if __name__ == "__main__":
    usage = "usage: %prog [options] msi_file1 msi_file2 ..."
    parser = OptionParser(usage=usage, version="1.0 2019-4-14")
    parser.description = 'Introduction: calculate Theil-Sen Regression of landsat time series'

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
