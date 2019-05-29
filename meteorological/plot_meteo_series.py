#!/usr/bin/env python
# Filename: plot_snow_timeseries.py
"""
introduction: plot the time series of air temperature

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 3 May, 2019
"""

import sys,os
from optparse import OptionParser
import rasterio
import numpy as np

# import pandas as pd # read and write excel files

HOME = os.path.expanduser('~')
codes_dir2 = HOME + '/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)


import basic_src.io_function as io_function

import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import basic_src.RSImage as RSImage
from basic_src.RSImage import RSImageclass
import csv
# from msi_landsat8 import get_band_names  # get_band_names(img_path)

import pandas as pd
import seaborn as sns

import statsmodels.api as sm    # for linear regression

import matplotlib.ticker as ticker

def read_data_series(station_no, txt_list):
    '''
    read data from txt files
    :param station_no: station number, string type
    :param txt_list: data list
    :return: date_list, record1_list, record2_list, record3_list
    '''

    data_str_list = []
    first_record_list = []
    sec_record_list =[]
    third_record_list = []

    for txt in txt_list:
        with open(txt,'r') as obj:
            # one txt contains the data for one month
            str_lines = obj.readlines()

            # filter the lines by station
            station_lines = [ str_line for str_line in str_lines if station_no in str_line ]

            for str_line in station_lines:
                tmp_array = str_line.split()
                # print(str_line)
                # if '-'.join(tmp_array[4:6]) != '2014-10':    # for comparison, test
                #     continue
                data_str_list.append('-'.join(tmp_array[4:7]))  # year, month, day
                first_record_list.append(float(tmp_array[7]))
                sec_record_list.append(float(tmp_array[8]))
                third_record_list.append(float(tmp_array[9]))

    return data_str_list,first_record_list,sec_record_list,third_record_list


def read_temperature_series(data_folder, station_no):

    txt_list = io_function.get_file_list_by_ext('.TXT', data_folder,bsub_folder=False)
    txt_list_2 = io_function.get_file_list_by_ext('.txt', data_folder, bsub_folder=False)
    txt_list.extend(txt_list_2)

    # read daily mean, max, and min temperature
    date_list, mean_tem, max_tem, min_tem =  read_data_series(station_no, txt_list)

    # convert to degree
    mean_tem = [tem/10.0 for tem in mean_tem ]  # ignore  if tem < 1000 and tem > -1000
    max_tem = [tem / 10.0 for tem in max_tem]
    min_tem = [tem / 10.0 for tem in min_tem]

    return date_list, mean_tem, max_tem, min_tem

def read_precipitation_series(data_folder, station_no):

    txt_list = io_function.get_file_list_by_ext('.TXT', data_folder,bsub_folder=False)
    txt_list_2 = io_function.get_file_list_by_ext('.txt', data_folder, bsub_folder=False)
    txt_list.extend(txt_list_2)

    # read daily mean, max, and min precipitation
    date_list, mean_pre, max_pre, min_pre =  read_data_series(station_no, txt_list)

    # convert to mm
    mean_pre = [tem / 10.0 for tem in mean_pre  ]  #
    max_pre = [tem / 10.0 for tem in max_pre   ]
    min_pre = [tem / 10.0 for tem in min_pre ]

    date_list_new = []
    mean_pre_new = []
    max_pre_new = []
    min_pre_new = []
    # remove outliers
    for date, pre1, pre2, pre3 in zip(date_list,mean_pre,max_pre,min_pre):
        if pre1 > 1000 or pre2 > 1000 or pre3 > 1000:
            continue
        date_list_new.append(date)
        mean_pre_new.append(pre1)
        max_pre_new.append(pre2)
        min_pre_new.append(pre3)

    return date_list_new, mean_pre_new, max_pre_new, min_pre_new

def save2txt(time_series, output,column_name=None):

    with open(output,'w') as txt_obj:
        # print('start')
        x_list = time_series.index.values
        y_list = time_series.values
        # print(x_list)
        # print(y_list)
        for index, value in zip(x_list,y_list):
            txt_obj.writelines('%s : %f \n'%(str(index),value))


    pass

def plot_air_tem_series(data_folder, station_no):
    '''
    plot figures of air temperature
    :param data_folder:
    :param station_no:
    :return:
    '''

    if 'tem' not in data_folder:
        raise ValueError('this is not the folder of air temperature')

    date_list, mean_tem, max_tem, min_tem = read_temperature_series(data_folder, station_no)
    # print(date_list)

    # plot time series data
    data = {'date': date_list, 'mean_air_tem': mean_tem, 'max_air_tem': max_tem, 'min_air_tem': min_tem}
    tem_series = pd.DataFrame(data, columns=['date', 'mean_air_tem', 'max_air_tem', 'min_air_tem'])
    #
    # convert to datetime format
    tem_series['date'] = pd.to_datetime(tem_series['date'], format='%Y-%m-%d')

    # set DatetimeIndex
    tem_series = tem_series.set_index('date')

    # sort, seem not necessary
    tem_series = tem_series.sort_index()

    # set date_range
    tem_series = tem_series["2002-01-01":"2012-12-31"]

    # Add columns with year, month, and weekday name
    # msi_series['Year'] = msi_series.index.year
    # msi_series['Month'] = msi_series.index.month
    #
    # print(msi_series.head(10))
    # print(msi_series.shape)
    # print(msi_series.dtypes)

    # print(tem_series.describe())
    # print(tem_series.columns)

    # scatter plot between each variables
    # sns.pairplot(tem_series)
    # plt.show()

    # plot historgram
    # sns.distplot(tem_series['max_air_tem'])  # mean_air_tem has an outlier
    # plt.show()

    # Correlation
    # print(tem_series.corr())

    #####################
    ## linear regression
    X = tem_series.index.strftime("%Y%m%d").astype(int)
    # X = sm.add_constant(X)        # add a constant
    # y = tem_series['max_air_tem']
    y = tem_series['mean_air_tem']
    # Note the difference in argument order
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)  # make the predictions by the model
    # Print out the statistics
    print(model.summary())

    tem_series['predictions'] = predictions

    # plot the air temperature and linear trend together.
    cols_plot = ['mean_air_tem', 'predictions'] # mean_air_tem  max_air_tem, min_air_tem
    # ylim_list = [(-20,30), (-20, 30) ]
    axes = tem_series[cols_plot].plot(x=tem_series.index, y=cols_plot, marker='.', alpha=0.9, linestyle='None',
                                      figsize=(21, 8))  # linewidth=0.5
    axes.set_xlabel('Years', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    # plt.title('Air temperature',fontsize=16)

    # axes.set_ylabel('Daily max air temperature')
    axes.set_ylabel('Daily mean air temperature',fontsize=16)
    # axes.set_ylabel('Daily min air temperature',fontsize=16)
    # axes.set_ylim(-20,30)
    axes.set_ylim(-25, 15)
    # axes.set_ylim(-35, 10)
    # axes.set_ylim(-5,5)
    print(max(predictions) - min(predictions))

    # cols_plot = ['mean_air_tem', 'max_air_tem','min_air_tem']
    # ylim_list = [(-30,20), (-20,30), (-35,10) ]
    # axes = tem_series[cols_plot].plot(marker='.', alpha=0.9,linestyle='None' , figsize=(21, 16), subplots=True)  # linewidth=0.5
    # for idx,ax in enumerate(axes):
    #     ax.set_ylabel(cols_plot[idx])
    #     ax.set_ylim(ylim_list[idx])

    # df.set_index('date').plot()
    # df.plot(x='date', y='brightness')
    # plt.show()
    output = 'fig_' + str(np.random.randint(1, 10000)) + '.png'
    plt.savefig(output, bbox_inches="tight")  # dpi=200, ,dpi=300

    # sns.regplot(x='date', y="mean_air_tem", data=tem_series);


def plot_gst_tem_series(data_folder, station_no):

    if 'gst' not in data_folder:
        raise ValueError('this is not the folder of ground surface temperature')

    date_list, mean_tem, max_tem, min_tem = read_temperature_series(data_folder, station_no)

    # plot time series data
    data = {'date': date_list, 'mean_ground_tem': mean_tem, 'max_ground_tem': max_tem, 'min_ground_tem': min_tem}
    tem_series = pd.DataFrame(data, columns=['date', 'mean_ground_tem', 'max_ground_tem', 'min_ground_tem'])
    #
    # convert to datetime format
    tem_series['date'] = pd.to_datetime(tem_series['date'], format='%Y-%m-%d')

    # set DatetimeIndex
    tem_series = tem_series.set_index('date')

    # sort, seem not necessary
    tem_series = tem_series.sort_index()

    # set date_range
    tem_series = tem_series["2002-01-01":"2012-12-31"]

    # Add columns with year, month, and weekday name
    # msi_series['Year'] = msi_series.index.year
    # msi_series['Month'] = msi_series.index.month
    #
    # print(msi_series.head(10))
    # print(msi_series.shape)
    # print(msi_series.dtypes)

    # print(tem_series.describe())
    # print(tem_series.columns)

    # scatter plot between each variables
    # sns.pairplot(tem_series)
    # plt.show()

    # plot historgram
    # sns.distplot(tem_series['max_air_tem'])  # mean_air_tem has an outlier
    # plt.show()

    # Correlation
    # print(tem_series.corr())

    #####################
    ## linear regression
    # X = tem_series.index.strftime("%Y%m%d").astype(int)
    # # X = sm.add_constant(X)        # add a constant
    # y = tem_series['max_air_tem']
    # # Note the difference in argument order
    # model = sm.OLS(y, X).fit()
    # predictions = model.predict(X)  # make the predictions by the model
    # # Print out the statistics
    # print(model.summary())
    #
    # tem_series['predictions'] = predictions

    ######## plot the air temperature and linear trend together.
    # cols_plot = ['max_air_tem', 'predictions']
    # # ylim_list = [(-20,30), (-35,10) ]
    # axes = tem_series[cols_plot].plot(x=tem_series.index, y=cols_plot, marker='.', alpha=0.9, linestyle='None',
    #                                   figsize=(21, 8))  # linewidth=0.5
    # axes.set_ylabel('Daily max air temperature')
    # # axes.set_ylim(-5,5)
    # print(max(predictions) - min(predictions))

    ##########
    cols_plot = ['mean_ground_tem', 'max_ground_tem','min_ground_tem']
    ylim_list = [(-30,25), (-15,70), (-35,10) ]
    axes = tem_series[cols_plot].plot(marker='.', alpha=0.9,linestyle='None' , figsize=(21, 16), subplots=True)  # linewidth=0.5
    for idx,ax in enumerate(axes):
        ax.set_ylabel(cols_plot[idx])
        ax.set_ylim(ylim_list[idx])

    # df.set_index('date').plot()
    # df.plot(x='date', y='brightness')
    # plt.show()
    output = 'fig_' + str(np.random.randint(1, 10000)) + '.png'
    plt.savefig(output, bbox_inches="tight")  # dpi=200, ,dpi=300

    # sns.regplot(x='date', y="mean_air_tem", data=tem_series);


def plot_pre_series(data_folder, station_no):

    # if 'pre' not in data_folder:
    #     raise ValueError('this is not the folder of Precipitation')

    date_list, pre20_8, pre8_20, pre20_20 = read_precipitation_series(data_folder, station_no)

    # plot time series data
    data = {'date': date_list, 'pre20_8': pre20_8, 'pre8_20': pre8_20, 'pre20_20': pre20_20}
    tem_series = pd.DataFrame(data, columns=['date', 'pre20_8', 'pre8_20', 'pre20_20'])
    #
    # convert to datetime format
    tem_series['date'] = pd.to_datetime(tem_series['date'], format='%Y-%m-%d')

    # set DatetimeIndex
    tem_series = tem_series.set_index('date')

    # sort, seem not necessary
    tem_series = tem_series.sort_index()

    # set date_range
    tem_series = tem_series["2002-01-01":"2012-12-31"]

    # Add columns with year, month, and weekday name
    # msi_series['Year'] = msi_series.index.year
    # msi_series['Month'] = msi_series.index.month
    #
    # print(msi_series.head(10))
    # print(msi_series.shape)
    # print(msi_series.dtypes)

    print(tem_series.describe())
    print(tem_series.columns)

    # scatter plot between each variables
    # sns.pairplot(tem_series)
    # plt.show()

    # plot historgram
    # sns.distplot(tem_series['max_air_tem'])  # mean_air_tem has an outlier
    # plt.show()

    # Correlation
    # print(tem_series.corr())

    #####################
    ## linear regression
    # X = tem_series.index.strftime("%Y%m%d").astype(int)
    # # X = sm.add_constant(X)        # add a constant
    # y = tem_series['max_air_tem']
    # # Note the difference in argument order
    # model = sm.OLS(y, X).fit()
    # predictions = model.predict(X)  # make the predictions by the model
    # # Print out the statistics
    # print(model.summary())
    #
    # tem_series['predictions'] = predictions

    ######## plot the air temperature and linear trend together.
    # cols_plot = ['max_air_tem', 'predictions']
    # # ylim_list = [(-20,30), (-35,10) ]
    # axes = tem_series[cols_plot].plot(x=tem_series.index, y=cols_plot, marker='.', alpha=0.9, linestyle='None',
    #                                   figsize=(21, 8))  # linewidth=0.5
    # axes.set_ylabel('Daily max air temperature')
    # # axes.set_ylim(-5,5)
    # print(max(predictions) - min(predictions))

    ##########
    # cols_plot = ['pre20_8', 'pre8_20','pre20_20']
    # ylim_list = [(0,50), (0,50), (0,50) ]
    # axes = tem_series[cols_plot].plot(marker='.', alpha=0.9,linestyle='None' , figsize=(21, 16), subplots=True)  # linewidth=0.5
    # for idx,ax in enumerate(axes):
    #     ax.set_ylabel(cols_plot[idx])
    #     ax.set_ylim(ylim_list[idx])


    ######
    # plot monthly data
    # tem_series['Year'] = tem_series.index.year
    # tem_series['Month'] = tem_series.index.month
    # cols_plot = ['pre20_20']
    # year_month_s_days = tem_series.groupby(['Year', 'Month'])['pre20_20'].apply(sum)
    # # axes = year_month_s_days.plot(x=tem_series.index, y=cols_plot, marker='.', alpha=0.9, linestyle='None', figsize=(21, 16), subplots=True)
    # axes = year_month_s_days.plot.bar(x=tem_series.index, y=cols_plot, figsize=(21, 16), subplots=True)
    #
    # interval = 6 # very six tick
    # axes[0].set_ylabel('Monthly precipitation (mm)', fontsize=20)
    # axes[0].set_xlabel('Year, Month', fontsize=20)
    # plt.tick_params(axis='both', which='major', labelsize=20,length=8)
    # axes[0].xaxis.set_major_locator(ticker.MultipleLocator(interval))
    #
    # seq = axes[0].xaxis.get_major_formatter().seq
    # axes[0].xaxis.set_major_formatter(ticker.FixedFormatter([""] + seq[::interval]))
    # # plt.title('Monthly precipitation',fontsize=20)
    # axes[0].set_title("")

    ######
    # # plot yearly data
    # yearly_pre = tem_series.groupby(['Year'])['pre20_20'].apply(sum)
    # axes = yearly_pre.plot.bar(x=tem_series.index, y=cols_plot, figsize=(21, 16), subplots=True)
    #
    # # axes[0].set_xticklabels('x',fontsize=16)
    # axes[0].set_ylabel('Annual precipitation (mm)', fontsize=20)
    # axes[0].set_xlabel('Year', fontsize=20)
    # plt.tick_params(axis='both', which='major', labelsize=20)
    # # plt.title('Annual precipitation',fontsize=16)
    # axes[0].set_title("")



    #################
    ## plot yearly, monthly together
    tem_series['Year'] = tem_series.index.year
    tem_series['Month'] = tem_series.index.month
    cols_plot = ['pre20_20']
    year_month_s_days = tem_series.groupby(['Year', 'Month'])['pre20_20'].apply(sum)
    yearly_pre = tem_series.groupby(['Year'])['pre20_20'].apply(sum)
    # axes = year_month_s_days.plot(x=tem_series.index, y=cols_plot, marker='.', alpha=0.9, linestyle='None', figsize=(21, 16), subplots=True)
    # axes = year_month_s_days.plot.bar(x=tem_series.index, y=cols_plot, figsize=(21, 16), subplots=True)

    # save2txt(year_month_s_days, 'year_month_s_days.txt')
    save2txt(yearly_pre, 'yearly_pre.txt')

    # print(yearly_pre)
    # plt.scatter(yearly_pre.index, yearly_pre)
    # tem_series.plot(style='.')

    # df = yearly_pre.to_frame()
    # print(df)
    # df.plot(kind='scatter', x='Year', y='pre20_20')

    # print(year_month_s_days.index)
    # print(year_month_s_days)
    # plt.scatter(year_month_s_days.index, year_month_s_days)
    # plt.show()

    # axes  =  pd.DataFrame.plot.scatter(tem_series.index, year_month_s_days)
    #
    # interval = 6 # very six tick
    # axes[0].set_ylabel('Monthly precipitation (mm)', fontsize=20)
    # axes[0].set_xlabel('Year, Month', fontsize=20)
    # plt.tick_params(axis='both', which='major', labelsize=20,length=8)
    # axes[0].xaxis.set_major_locator(ticker.MultipleLocator(interval))
    #
    # seq = axes[0].xaxis.get_major_formatter().seq
    # axes[0].xaxis.set_major_formatter(ticker.FixedFormatter([""] + seq[::interval]))
    # # plt.title('Monthly precipitation',fontsize=20)
    # axes[0].set_title("")


    # df.set_index('date').plot()
    # df.plot(x='date', y='brightness')
    # plt.show()
    # output = 'fig_' + str(np.random.randint(1, 10000)) + '.png'
    # plt.savefig(output, bbox_inches="tight",dpi=300)  # dpi=200, ,dpi=300

    # sns.regplot(x='date', y="mean_air_tem", data=tem_series);


def main(options, args):

    data_folder = args[0]

    station_no = options.station

    data_type = options.data_type

    # air temperature
    if data_type.upper() == 'TEM':
        plot_air_tem_series(data_folder, station_no)
    elif data_type.upper() == 'PRE':
        plot_pre_series(data_folder, station_no)
    elif data_type.upper() == 'GST':
        plot_gst_tem_series(data_folder, station_no)
    else:
        print('%s not has been supported'%data_type)



if __name__ == "__main__":
    usage = "usage: %prog [options] data_folder"
    parser = OptionParser(usage=usage, version="1.0 2019-4-14")
    parser.description = 'Introduction: plot the time series of landsat data'

    parser.add_option("-o", "--output",
                      action="store", dest="output",
                      help="the output file path")

    parser.add_option("-s","--station",
                      action="store", dest="station",
                      help="the station number")

    parser.add_option("-d","--data_type",
                      action="store", dest="data_type",
                      help="data_tyep, including: pre, tem, gst (need to add more in the future)")

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
