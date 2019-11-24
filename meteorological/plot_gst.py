#!/usr/bin/env python
# Filename: plot_gst.py
"""
introduction: plot the time series of ground surface temperature

authors: Hu Yan
email:yan_hu@hotmail.com
add time: 23 Oct, 2019
"""

import sys,os
from optparse import OptionParser
import rasterio
import numpy as np
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

# import pandas as pd # read and write excel files

HOME = os.path.expanduser('~')
codes_dir2 = HOME + '/codes/DeeplabforRS'
sys.path.insert(0, codes_dir2)

import matplotlib.pyplot as plt

def read_daily_gst(txt_path):

    date_list = []
    gst_list = []
    with open(txt_path,'r') as f_obj:
        lines = f_obj.readlines()

        for line in lines:
            # str_list = line.split('*')
            ## for ten-year_mean
            str_list = line.split(':')
            date = str_list[0].strip()
            gs_tem = float(str_list[1])

            date_list.append(date)
            gst_list.append(gs_tem)
    return date_list, gst_list

def read_monthly_gst(txt_path):

    month_list = []
    gst_list = []
    with open(txt_path,'r') as f_obj:
        lines = f_obj.readlines()

        for line in lines:
            str_list = line.split(':')
            month = str_list[0].strip()
            gs_tem = float(str_list[1])

            month_list.append(month)
            gst_list.append(gs_tem)
    return month_list, gst_list

    pass
# read data

date_list, daily_gst = read_daily_gst('daily_mean_gst.txt')
#
# for idx in range(0, len(daily_gst)):
#     gst =  daily_gst[idx]
#     # remove abnormal value
#     if gst > 100 or gst < -100:
#         daily_gst[idx] = daily_gst[idx-1]
#         print('One abnormal value is replaced by the value of previous day: %f'%gst)
#
fig = plt.figure(figsize=(12, 4))
# #
x_date = range(0,len(date_list))
#
#
## draw original data
plt.plot(x_date, daily_gst,'k+',linewidth=0.3, markersize=3, label='2008-2018 daily mean ground surface temperature')

## draw tem=0
plt.axhline(y=0, color='grey', linestyle='--')

##
plt.legend()
plt.ylim([-25,25])
plt.xlabel('Month, Day')
plt.ylabel('Ground surface temperature ($^\circ$C)')
#
## for the ten-year-mean data
x_tick_loc = x_date[::30]
x_labels = date_list[::30]
x_labels = [item.strip('(').strip(')') for item in x_labels]
#
plt.xticks(x_tick_loc)
plt.gca().set_xticklabels(x_labels)
plt.tick_params(axis='x', which='minor', length=3)
plt.tick_params(axis='x', which='major', length=5, rotation=90)

# plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
# ## set up the axis label
# x_tick_loc = []
# x_labels = []
# for loc, date in zip(x_date, date_list):
#      if "-01-01" in date:
#          x_tick_loc.append(loc)
#          x_labels.append(date.split('-')[0] + '-' + date.split('-')[1])
#      if "-04-01" in date:
#          x_tick_loc.append(loc)
#          x_labels.append(date.split('-')[0] + '-' + date.split('-')[1])
#      if "-07-01" in date:
#          x_tick_loc.append(loc)
#          x_labels.append(date.split('-')[0] + '-' + date.split('-')[1])
#      if "-10-01" in date:
#          x_tick_loc.append(loc)
#          x_labels.append(date.split('-')[0] + '-' + date.split('-')[1])
#
# print(x_tick_loc)
# print(x_labels)
#
# #
# plt.xticks(x_tick_loc)
# plt.gca().set_xticklabels(x_labels)
# plt.tick_params(axis='x', which='minor',length=3)
# plt.tick_params(axis='x', which='major', length=5, rotation=90) #labelsize=10
#
plt.gcf().subplots_adjust(bottom=0.25)
# plt.show()
# #
output = 'ten-year_daily_mean_gst.png'
plt.savefig(HOME + '/LAB/' + output, bbox_inches="tight",dpi=300)  # dpi=200, ,dpi=300

################
## plot monthly gst
# month_list, monthly_gst = read_monthly_gst('monthly_mean_gst.txt')
#
# fig = plt.figure(figsize=(12, 4))
#
# x_month = range(0,len(month_list))
#
# # draw
# plt.plot(x_month, monthly_gst, '-k+', linewidth=1, markersize=6, label='Monthly mean ground surface temperature')
# plt.axhline(y=0, color='grey', linestyle='--')
#
# plt.legend()
# plt.xlabel('Year, Month')
# plt.ylabel('Mean ground surface temperature ($^\circ$C)')
#
# x_tick_loc = x_month[::6]
# x_labels = month_list[::6]
# x_labels = [item.strip('(').strip(')') for item in x_labels]
#
# plt.xticks(x_tick_loc)
# plt.gca().set_xticklabels(x_labels)
# plt.tick_params(axis='x', which='minor', length=3)
# plt.tick_params(axis='x', which='major', length=5, rotation=90)
#
# plt.gca().xaxis.set_minor_locator(MultipleLocator(1))

# plt.gcf().subplots_adjust(bottom=0.25)
# plt.show()

# output = 'monthly_ground_surface_temperature.png'
# plt.savefig(HOME + '/LAB/' + output, bbox_inches="tight",dpi=300)  # dpi=200, ,dpi=300