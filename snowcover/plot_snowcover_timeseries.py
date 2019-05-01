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
import matplotlib
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
    values_f = [int(item) for item in values]
    series_value.extend(values_f)


    # read other data
    dir_path = os.path.dirname(snow_file)
    basename = os.path.basename(snow_file)
    tmp_str =  basename.split('_')
    # read NDSI_Snow_Cover_Class
    NDSI_Snow_Cover_Class_path = os.path.join(dir_path,
                                '_'.join(tmp_str[:2]) + '_NDSI_Snow_Cover_Class_'+tmp_str[2])
    # print(NDSI_Snow_Cover_Class_path)
    Class_values = RSImage.get_image_location_value_list(NDSI_Snow_Cover_Class_path, x, y, xy_srs)
    Class_values_f = [int(item) for item in Class_values]

    # read Snow_Albedo_Daily_Tile_Class
    Snow_Albedo_Daily_Tile_Class = os.path.join(dir_path,
                                '_'.join(tmp_str[:2]) + '_Snow_Albedo_Daily_Tile_Class_'+tmp_str[2])
    # print(Snow_Albedo_Daily_Tile_Class)
    Tile_Class_values = RSImage.get_image_location_value_list(Snow_Albedo_Daily_Tile_Class, x, y, xy_srs)
    Tile_Class_values_f = [int(item) for item in Tile_Class_values]

    if len(series_value) != len(Class_values_f):
        raise ValueError('the length is inconsistent')

    if len(series_value) != len(Tile_Class_values_f):
        raise ValueError('the length is inconsistent')

    return series_value, Class_values_f, Tile_Class_values_f

def get_snow_cover(data_pd):
    '''
    merge the daily snow cover data from MOD and MYD
    :param data_pd: pandas dataframe, using date as index, sorted by date
    :return: pandas dataframe, only one record for each day
    '''
    # divide to group based on date
    # data_group = data_pd.groupby(data_pd.index) #.aggregate(func, axis=0, *args, **kwargs)[source]

    # remove no_decision, missing data
    # data_pd= data_pd[data_pd.Class != 201]
    # data_pd = data_pd[data_pd.Class != 200]

    date_list = []

    land_cover = [] # (1, 2, 3) snow, others (land, inland water or others), unknown (due to cloud or missing data)

    # check the data date by date.
    for date_value in set(data_pd.index):  # use set to get unique dates
        row = data_pd.loc[date_value,:].values  # get a row by date index

        date_list.append(date_value)
        # for this date, only have one record
        if len(row) == 3:    # 1D list
            # MODIS_snow_cover_list.append(row[0])
            # Class_values_list.append(row[1])
            # Tile_Class_values_list.append(row[2])
            if row[0] > 0 and row[1]==0 and row[2]==0:
                land_cover.append(1)    # this is snow
            elif row[2] in [125,137,139]:  # land, inland water, ocean
                land_cover.append(2)  # others land cover
            else:
                land_cover.append(3)    # unknow due to cloud, missing data and so on
            continue
        # 2D list, for the date has two records
        saved_row = row[0]
        if (row[0][0] > 0 and row[0][1]==0 and row[0][2]==0) or (row[1][0] > 0 and row[1][1]==0 and row[1][2]==0):
            land_cover.append(1)    # this is snow
        elif row[0][2] in [125,137,139] or row[1][2] in [125,137,139]:  # land, inland water, ocean
            land_cover.append(2)  # others land cover
        else:
            land_cover.append(3)    # unknow due to cloud, missing data and so on

    # create a new dataframe
    data = {'date': date_list, 'land_cover': land_cover}
    msi_series = pd.DataFrame(data, columns=['date', 'land_cover'])
    #
    # convert to datetime format
    msi_series['date'] = pd.to_datetime(msi_series['date'])

    # set DatetimeIndex
    msi_series = msi_series.set_index('date')

    # sort, seem not necessary
    msi_series = msi_series.sort_index()

    return msi_series

def fill_land_cover_series(land_cover):
    '''
    fill the unknown value based on adjacent values
    :param land_cover: dataframe of land cover of on pixel, in time order
    :return:
    '''

    land_type_array = land_cover.loc[:, 'land_cover'].values

    land_type_array_copy = np.copy(land_type_array)

    total_count = len(land_type_array)

    # fill the unknow values, based on the nearest (the past is first)
    for idx,land_type in enumerate(land_type_array):
        if land_type != 3:  # 1 or 2, that is snow or other land cover
            continue

        for win_size in range(1,30):
            if (idx - win_size)  >= 0 and land_type_array[idx - win_size] != 3:
                land_type_array_copy[idx] = land_type_array[idx - win_size]
                break
            elif (idx + win_size) < total_count and land_type_array[idx + win_size] != 3:
                land_type_array_copy[idx] = land_type_array[idx + win_size]
                break
        if land_type_array_copy[idx] == 3:
            raise ValueError('Can not find its adjacent value in a window of 30')

    result = land_cover.copy(deep=True)

    result.loc[:, 'land_cover'] = land_type_array_copy

    return result


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
    time_series_snow, Class_values, Tile_Class_values = read_time_series_snow(mod_snow_file,x,y, xy_srs)
    if len(date_str_list) != len(time_series_snow):
        raise ValueError('the length of snow and date_str is different')

    # read snow cover from MYD10A1 product
    myd_date_str_list = get_date_string_list(myd_snow_file)
    myd_time_series_snow,myd_Class_values, myd_Tile_Class_values = read_time_series_snow(myd_snow_file,x,y, xy_srs)
    if len(myd_date_str_list) != len(myd_time_series_snow):
        raise ValueError('the length of snow and date_str is different')


    date_str_list.extend(myd_date_str_list)
    time_series_snow.extend(myd_time_series_snow)

    Class_values.extend(myd_Class_values)
    Tile_Class_values.extend(myd_Tile_Class_values)


    data = {'date':date_str_list, 'MODIS_snow_cover':time_series_snow,'Class':Class_values,'Tile_Class':Tile_Class_values }
    msi_series = pd.DataFrame(data, columns=['date', 'MODIS_snow_cover', 'Class', 'Tile_Class'])
    #
    # convert to datetime format
    msi_series['date'] = pd.to_datetime(msi_series['date'], format='%Y_%m_%d')

    # set DatetimeIndex
    msi_series = msi_series.set_index('date')

    # sort, seem not necessary
    msi_series = msi_series.sort_index()

    ## check the data date by date.
    # for date_value in set(msi_series.index):  # use set to get unique dates
    #     row = msi_series.loc[date_value,:].values  # get a row by date index
    #     if len(row) == 3:    # 1D list
    #         continue
    #     # 2D list
    #     if np.array_equal(row[0],row[1]):
    #         continue
    #     else:
    #         print(date_value,row[0], row[1])

    # remove nan, if one cloumn is nan, other also nan
    msi_series = msi_series.dropna(how='any')
    print('msi_series row count:',len(msi_series.index))

    # # remove duplicate date
    # aaaa = msi_series.groupby(msi_series.index).max() # which column it used for calculating the max value?
    # print('msi_series row count:', len(msi_series.index))
    # print('aaaa row count:', len(aaaa.index))

    landcover_series = get_snow_cover(msi_series)
    landcover_series.to_excel('landcover_series.xlsx')

    # fill the unknow values
    landcover_series_filled =  fill_land_cover_series(landcover_series)

    # add
    landcover_series_filled['Year'] = landcover_series_filled.index.year
    landcover_series_filled['Month'] = landcover_series_filled.index.month

    landcover_series_filled.to_excel('landcover_series_filled.xlsx')

    # only keep the records with snow cover
    snow_series = landcover_series_filled[landcover_series_filled.land_cover == 1]
    snow_series.to_excel('snow_series.xlsx')
    year_month_s_days = snow_series.groupby(['Year', 'Month'])['land_cover'].apply(sum)


    year_month_s_days.to_frame(name='year_month_snow_days').to_excel('snow_series_year_month.xlsx')
    # print(year_month_s_days.head(1000))

    year_s_days = snow_series.groupby(['Year'])['land_cover'].apply(sum)
    print(year_s_days.head(1000))

    # test = 1

    # set date_range
    # msi_series = msi_series["2012-01-01":"2012-12-01"]


    # Add columns with year, month, and weekday name
    # msi_series['Year'] = msi_series.index.year
    # msi_series['Month'] = msi_series.index.month

    # print(msi_series.head(100))
    # print(msi_series.shape)
    # print(msi_series.dtypes)

    # msi_series.to_excel("msi_series.xlsx")
    # aaaa.to_excel("aaaa.xlsx")

    # print(msi_series.loc['2001-06'])

    # Use seaborn style defaults and set the default figure size
    # sns.set(rc={'figure.figsize': (21, 4)})
    # msi_series['brightness'].plot(marker='.',linestyle='None') #linewidth=1.5

    # # plot snow value
    # cols_plot = ['MODIS_snow_cover']
    # ylim_list = [(1,100) ]
    # axes = msi_series[cols_plot].plot(marker='.', alpha=0.9,linestyle='None',
    #                                   figsize=(21, 16), subplots=True)  # linewidth=0.5
    # for idx,ax in enumerate(axes):
    #     ax.set_ylabel(cols_plot[idx])
    #     ax.set_ylim(ylim_list[idx])
    #
    # # df.set_index('date').plot()
    # # df.plot(x='date', y='brightness')
    # # plt.show()
    # output='fig_'+str(np.random.randint(1,10000))+'.png'
    # # plt.savefig(output,bbox_inches="tight") # dpi=200, ,dpi=300

    ## plot land cover
    # cols_plot = ['land_cover']
    # ylim_list = [(0,4) ]
    # axes = landcover_series[cols_plot].plot(marker=',', alpha=0.9,linestyle='None',
    #                                   figsize=(21, 4), subplots=True)  # linewidth=0.5
    # for idx,ax in enumerate(axes):
    #     ax.set_ylabel(cols_plot[idx])
    #     ax.set_ylim(ylim_list[idx])
    #
    # # df.set_index('date').plot()
    # # df.plot(x='date', y='brightness')
    # # plt.show()
    # output='fig_'+str(np.random.randint(1,10000))+'.png'
    # plt.savefig(output,bbox_inches="tight") # dpi=200, ,dpi=300


    ## plot snow series by month
    # axes = year_month_s_days.plot(marker='.', alpha=0.9,linestyle='None', figsize=(21, 2))
    ## axes.xaxis.set_minor_locator(matplo tlib.dates.WeekdayLocator(byweekday=(1),interval=1))
    ## axes.xaxis.set_minor_locator(matplotlib.dates.MonthLocator())
    # output='fig_'+str(np.random.randint(1,10000))+'.png'
    # plt.savefig(output,bbox_inches="tight") # dpi=200, ,dpi=300







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
