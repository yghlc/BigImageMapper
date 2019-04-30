#!/usr/bin/env python
# Filename: plot_snow_timeseries.py
"""
introduction: plot the time series of snow cover from MODIS

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 30 April, 2019
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
import csv
# from msi_landsat8 import get_band_names  # get_band_names(img_path)

import pandas as pd

# use seaborn styling for our plots
import seaborn as sns

def remove_nan_value(msi_str_list, image_date_list):
    # remove non-data "nan"
    tmp_list = [(float(item), date_str) for item,date_str in zip(msi_str_list,image_date_list)  ] # if item != 'nan'

    msi_list = [item[0] for item in tmp_list]       # float
    date_list = [item[1] for item in tmp_list]      # string

    return msi_list, date_list

def get_date_string_list(file_name):
    """
    get the date string (%Y_%m_%d) stored in csv file
    :param img_files:
    :return: a list stored date strings
    """

    csv_file = os.path.splitext(file_name)[0]+'.csv'
    with open(csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for line_count, row in enumerate(csv_reader):
            if line_count == 0:
                continue
            date_str_list_str = row[1]
            date_str_list_str = date_str_list_str.strip('[').strip(']')
            date_str_list = date_str_list_str.split(',')
            date_str_list = [item.strip() for item in date_str_list ]
            return date_str_list



def read_time_series_snow(snow_file, x, y, xy_srs):
    '''
    read a time series of data from files for a specific point
    :param snow_file: a snow cover file
    :param x: x value
    :param y: y value
    :param xy_srs: the coordinate system of (x,y), the value is :pixel ,prj or lon_lat_wgs84
    :return:
    '''
    series_value = []


    values = RSImage.get_image_location_value_list(snow_file, x, y, xy_srs)
    # convert to float
    values_f = [float(item) for item in values]
    series_value.extend(values_f)

    return series_value



def main(options, args):

    mod_snow_file = args[0]
    myd_snow_file = args[1]

    # x = 10
    # y = 100

    # at least four decimal
    # x = 92.9123
    # y = 34.8485
    # x = 92.76071
    # y = 35.08546

    x = 92.80871
    y = 34.79564

    xy_srs = 'lon_lat_wgs84'  # pixel


    # read snow cover from  MOD10A1 product
    date_str_list = get_date_string_list(mod_snow_file)
    time_series_snow = read_time_series_snow(mod_snow_file,x,y, xy_srs)
    if len(date_str_list) != len(time_series_snow):
        raise ValueError('the length of snow and date_str is different')


    # read snow cover from MYD10A1 product
    myd_date_str_list = get_date_string_list(myd_snow_file)
    myd_time_series_snow = read_time_series_snow(myd_snow_file,x,y, xy_srs)
    if len(myd_date_str_list) != len(myd_time_series_snow):
        raise ValueError('the length of snow and date_str is different')


    date_str_list.extend(myd_date_str_list)
    time_series_snow.extend(myd_time_series_snow)

    data = {'date':date_str_list, 'MODIS_snow_cover':time_series_snow}
    msi_series = pd.DataFrame(data, columns=['date', 'MODIS_snow_cover'])
    #
    # convert to datetime format
    msi_series['date'] = pd.to_datetime(msi_series['date'], format='%Y_%m_%d')

    # set DatetimeIndex
    msi_series = msi_series.set_index('date')

    # sort, seem not necessary
    msi_series = msi_series.sort_index()

    # remove nan, if one cloumn is nan, other also nan
    msi_series =  msi_series.dropna(how='any')

    # set date_range
    msi_series = msi_series["2012-01-01":"2012-12-01"]


    # Add columns with year, month, and weekday name
    msi_series['Year'] = msi_series.index.year
    msi_series['Month'] = msi_series.index.month

    print(msi_series.head(100))
    print(msi_series.shape)
    print(msi_series.dtypes)

    # print(msi_series.loc['2001-06'])

    # Use seaborn style defaults and set the default figure size
    # sns.set(rc={'figure.figsize': (21, 4)})
    # msi_series['brightness'].plot(marker='.',linestyle='None') #linewidth=1.5

    cols_plot = ['MODIS_snow_cover']
    ylim_list = [(1,100) ]
    axes = msi_series[cols_plot].plot(marker='.', alpha=0.9,linestyle='None' , figsize=(21, 16), subplots=True)  # linewidth=0.5
    for idx,ax in enumerate(axes):
        ax.set_ylabel(cols_plot[idx])
        ax.set_ylim(ylim_list[idx])


    # df.set_index('date').plot()
    # df.plot(x='date', y='brightness')
    # plt.show()
    output='fig_'+str(np.random.randint(1,10000))+'.png'
    plt.savefig(output,bbox_inches="tight") # dpi=200, ,dpi=300






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
