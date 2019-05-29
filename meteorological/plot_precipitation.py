#!/usr/bin/env python
# Filename: plot_snow_timeseries.py
"""
introduction: plot the time series of air temperature

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 29 May, 2019
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

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import basic_src.RSImage as RSImage
from basic_src.RSImage import RSImageclass
import csv
# from msi_landsat8 import get_band_names  # get_band_names(img_path)

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

def read_year_pre(txt_path):
    year_list = []
    pre_list = []
    with open(txt_path,'r') as f_obj:
        lines = f_obj.readlines()

        for line in lines:
            str_list = line.split(':')
            year = int(str_list[0])
            pre = float(str_list[1])

            year_list.append(year)
            pre_list.append(pre)
    return year_list, pre_list



def read_monthly_pre(txt_path):

    month_list = []
    pre_list = []
    with open(txt_path,'r') as f_obj:
        lines = f_obj.readlines()

        for line in lines:
            str_list = line.split(':')
            month = str_list[0].strip()
            pre = float(str_list[1])

            month_list.append(month)
            pre_list.append(pre)
    return month_list, pre_list

    pass


# read data

year_list, year_pre = read_year_pre('yearly_pre.txt')
#
# print(year_list)
# print(year_pre)

month_list, month_pre = read_monthly_pre('year_month_s_days.txt')

# print(month_list)
# print(month_pre)

fig = plt.figure(figsize=(8, 4))

x_month = range(0,len(month_list))

y_year = x_month[6::12]

# draw
plt.plot(x_month,month_pre,'-k+',linewidth=1,markersize=6,label='Monthly precipitation')

# for year in y_year:
#     print(year)

plt.plot(y_year,year_pre,'-k*',linewidth=1,markersize=6,label='Annual precipitation')

for i, txt in enumerate(y_year):
    year_str = month_list[txt].split(',')[0].strip('(')
    x_off = 5
    y_off = 10

    if year_str in ['2003','2006','2007','2010','2011']:
        y_off = -25
    if year_str in ['2012']:
        y_off = -38

    plt.gca().annotate(year_str, (y_year[i], year_pre[i]), (y_year[i] - x_off,year_pre[i]+y_off ))


plt.legend()
plt.xlabel('Year, Month')
plt.ylabel('Precipitation (mm)')

# plt.tick_params(axis='both', which='major', labelsize=16)

x_tick_loc = x_month[::6]
x_labels = month_list[::6]
x_labels = [item.strip('(').strip(')') for item in x_labels]

plt.xticks(x_tick_loc)
plt.gca().set_xticklabels(x_labels)
plt.tick_params(axis='x', which='minor',length=3)
plt.tick_params(axis='x', which='major',length=5,rotation=90) #labelsize=10

plt.gca().xaxis.set_minor_locator(MultipleLocator(1))

# plt.show()

# output = 'fig_' + str(np.random.randint(1, 10000)) + '.png'
output = 'monthly_annual_precipitation.png'
plt.savefig(output, bbox_inches="tight",dpi=300)  # dpi=200, ,dpi=300


