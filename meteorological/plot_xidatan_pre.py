#!/usr/bin/env python
# Filename: plot_xidatan_pre.py
"""
introduction: plot time series of precipitation in Xidatan Station

authors: Hu Yan
email:yan_hu@hotmail.com
add time: 31 Oct, 2019
"""

import sys,os
from optparse import OptionParser
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_meteo_series import save2txt
from matplotlib.ticker import MultipleLocator

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

## read the excel data into dataframe
# data_file = '/home/yan_hu/LAB/xidatan_pre.xlsx'
data_file = '/home/yan_hu/LAB/xdt_monthly_pre.csv'
# pre_df = pd.read_excel(data_file, na_values='NAN')
#
# ## convert to datetime format and set it as index
# pre_df['date'] = pd.to_datetime(pre_df['Year']*1000 + pre_df['Day'], format='%Y%j')
# pre_df['date'] = pd.to_datetime(pre_df['date'], format='%Y-%m-%d')
# pre_df.set_index('date')
#
# ##plot monthly data
# pre_df['year'] = pre_df['date'].dt.year
# pre_df['month'] = pre_df['date'].dt.month
# cols_plot = ['Precipitation']
# monthly_data = pre_df.groupby(['year', 'month'])['Precipitation'].apply(sum)

## save to text
# save2txt(monthly_data, 'monthly_data.txt')

## read data from txt
# month_list, month_pre = read_monthly_pre('monthly_data.txt')
wdl_month_list, wdl_month_pre = read_monthly_pre('/home/yan_hu/LAB/wdl_monthly_pre.txt')
## read xdt data (from Ma)
xdt_month_pre = []
with open(data_file, 'r') as xdt_data:
    lines = xdt_data.readlines()
    for line in lines:
        pre = float(line.strip())
        xdt_month_pre.append(pre)

## draw monthly precipitation
fig = plt.figure(figsize=(12, 5))

x_month = range(0,len(wdl_month_list))

plt.plot(x_month, xdt_month_pre,'-k+', color='r', linewidth=1, markersize=6, label='Monthly precipitation in XDT')
plt.plot(x_month, wdl_month_pre, '-k+', color='b', linewidth=1, markersize=6, label='Monthly precipitation in WDL')

plt.legend()
plt.xlabel('Year, Month')
plt.ylabel('Precipitation (mm)')
plt.ylim(0, 200)

x_tick_loc = x_month[::6]
x_labels = wdl_month_list[::6]
x_labels = [item.strip('(').strip(')') for item in x_labels]

plt.xticks(x_tick_loc)
plt.gca().set_xticklabels(x_labels)
plt.tick_params(axis='x', which='minor',length=3)
plt.tick_params(axis='x', which='major',length=5,rotation=90)

plt.gca().xaxis.set_minor_locator(MultipleLocator(1))

plt.gcf().subplots_adjust(bottom=0.25)
# plt.show()

output = 'XDT_WDL_precipitation.png'
plt.savefig('/home/yan_hu/LAB/' + output, bbox_inches="tight",dpi=300)