#!/usr/bin/env python
# Filename: plot_landsat_timeseries.py
"""
introduction: plot the time series of landsat data

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 14 April, 2019
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
import matplotlib.pyplot as plt
import basic_src.RSImage as RSImage
from msi_landsat8 import get_band_names  # get_band_names(img_path)

import pandas as pd

# use seaborn styling for our plots
import seaborn as sns

def remove_nan_value(msi_str_list, image_date_list):
    # remove non-data "nan"
    tmp_list = [(float(item), date_str) for item,date_str in zip(msi_str_list,image_date_list)  ] # if item != 'nan'

    msi_list = [item[0] for item in tmp_list]       # float
    date_list = [item[1] for item in tmp_list]      # string

    return msi_list, date_list

def get_date_string_list(img_files):
    """
    get the date string (%Y-%m-%d) stored in files
    :param img_files:
    :return: a 2D list strong the date string
    """

    date_str_list = []
    for img_file in img_files:
        band_name_list = get_band_names(img_file)
        date_str = ['_'.join(item.split('_')[:1]) for item in band_name_list]  #date string format: (%Y-%m-%d)
        date_str_list.append(date_str)

    return date_str_list


def get_msi_file_list(all_files, product_name):
    '''
    get product file list from all the files
    :param all_files:
    :param product_name:
    :return:
    '''
    # get product list
    return [item for item in all_files if product_name in item]


def read_time_series(product_list, date_str_list, x, y, xy_srs):
    '''
    read a time series of data from files for a specific point
    :param product_list: date_str_list and product_list should match, i.e., each date_str are read from the file in product_list
    :param date_str_list: a 2D list containing date strings
    :param x: x value
    :param y: y value
    :param xy_srs: the coordinate system of (x,y), the value is :pixel ,prj or lon_lat_wgs84
    :return:
    '''

    if len(product_list) != len(date_str_list):
        raise ValueError('product files and date string not match')

    date_str = []
    series_value = []

    for img_file, date_string in zip(product_list,date_str_list):
        # read value string list
        values = RSImage.get_image_location_value_list(img_file, x, y, xy_srs)
        # convert to float
        values_f = [float(item) for item in values]

        date_str.extend(date_string)
        series_value.extend(values_f)

    return date_str, series_value


def plot_time_series_data(time_series):

    # from pandas import Series
    # from pandas import date_range
    # from numpy.random import randn

    # ts = Series(randn(1000), index=date_range('1/1/2000', periods=1000))
    #
    # ts = ts.cumsum()
    #
    # ts.plot()
    # plt.figure()
    # plt.figure()
    # ts.plot(style='k--', label='Series');
    # plt.legend()
    # time_series

    # plt.scatter()


    pass



def main(options, args):

    msi_files = args

    # x = 10
    # y = 100
    #
    x = 92.912
    y=34.848

    xy_srs = 'lon_lat_wgs84'  # pixel



    # read brightness values
    brightness_files = get_msi_file_list(msi_files,'brightness')
    # brightness_files = brightness_files[:2] # for test
    brightness_date_str_list = get_date_string_list(brightness_files)
    b_date_str, b_value = read_time_series(brightness_files,brightness_date_str_list,x,y, xy_srs)

    # read greenness
    greenness_files = get_msi_file_list(msi_files,'greenness')
    # greenness_files = greenness_files[:2] # for test
    greenness_date_str_list = get_date_string_list(greenness_files)
    g_date_str, g_value = read_time_series(greenness_files,greenness_date_str_list,x,y,xy_srs)

    # wetness
    wetness_files = get_msi_file_list(msi_files,'wetness')
    # wetness_files = wetness_files[:2] # for test
    wetness_date_str_list = get_date_string_list(wetness_files)
    w_date_str, w_value = read_time_series(wetness_files,wetness_date_str_list,x,y,xy_srs)

    # NDVI
    ndvi_files = get_msi_file_list(msi_files,'NDVI')
    # ndvi_files = ndvi_files[:2] # for test
    ndvi_date_str_list = get_date_string_list(ndvi_files)
    ndvi_date_str, ndvi_value = read_time_series(ndvi_files,ndvi_date_str_list,x,y,xy_srs)

    # NDWI
    ndwi_files = get_msi_file_list(msi_files,'NDWI')
    # ndwi_files = ndwi_files[:2] # for test
    ndwi_date_str_list = get_date_string_list(ndwi_files)
    ndwi_date_str, ndwi_value = read_time_series(ndwi_files,ndwi_date_str_list,x,y,xy_srs)

    # NDMI
    ndmi_files = get_msi_file_list(msi_files,'NDMI')
    # ndmi_files = ndmi_files[:2] # for test
    ndmi_date_str_list = get_date_string_list(ndmi_files)
    ndmi_date_str, ndmi_value = read_time_series(ndmi_files,ndmi_date_str_list,x,y,xy_srs)


    # check the sate_str are the same
    if b_date_str != g_date_str:
        raise ValueError('the date strings are different, "b_date_str != g_date_str"')
    # check the sate_str are the same
    if b_date_str != w_date_str:
        raise ValueError('the date strings are different, "b_date_str != w_date_str"')
    if b_date_str != ndvi_date_str:
        raise ValueError('the date strings are different, "b_date_str != ndvi_date_str"')
    if b_date_str != ndwi_date_str:
        raise ValueError('the date strings are different, "b_date_str != ndwi_date_str"')
    if b_date_str != ndmi_date_str:
        raise ValueError('the date strings are different, "b_date_str != ndmi_date_str"')

    data = {'date':b_date_str, 'brightness':b_value,'greenness':g_value,
            'wetness':w_value,'ndvi':ndvi_value,'ndwi':ndwi_value,'ndmi':ndmi_value}
    msi_series = pd.DataFrame(data, columns=['date', 'brightness','greenness','wetness','ndvi','ndwi','ndmi'])
    #
    # convert to datetime format
    msi_series['date'] = pd.to_datetime(msi_series['date'], format='%Y-%m-%d')

    # set DatetimeIndex
    msi_series = msi_series.set_index('date')


    # Add columns with year, month, and weekday name
    msi_series['Year'] = msi_series.index.year
    msi_series['Month'] = msi_series.index.month

    print(msi_series.head(10))
    print(msi_series.shape)
    print(msi_series.dtypes)

    # print(msi_series.loc['2001-06'])

    # Use seaborn style defaults and set the default figure size
    # sns.set(rc={'figure.figsize': (21, 4)})
    # msi_series['brightness'].plot(marker='.',linestyle='None') #linewidth=1.5

    cols_plot = ['greenness', 'brightness','wetness','ndvi','ndwi','ndmi']
    axes = msi_series[cols_plot].plot(marker='.', alpha=0.9, linestyle='None', figsize=(21, 16), subplots=True)
    for idx,ax in enumerate(axes):
        ax.set_ylabel(cols_plot[idx])

    # df.set_index('date').plot()
    # df.plot(x='date', y='brightness')
    # plt.show()
    output='fig_'+str(np.random.randint(1,10000))+'.png'
    plt.savefig(output)






if __name__ == "__main__":
    usage = "usage: %prog [options] msi_file1 msi_file2 ..."
    parser = OptionParser(usage=usage, version="1.0 2019-4-14")
    parser.description = 'Introduction: plot the time series of landsat data'

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
