#!/usr/bin/env python
# Filename: plot_air_tem.py
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

import statsmodels.api as sm    # for linear regression
import numpy as np
from scipy import optimize

from numpy import polyfit


def read_daily_tem(txt_path):

    date_list = []
    air_list = []
    with open(txt_path,'r') as f_obj:
        lines = f_obj.readlines()

        for line in lines:
            str_list = line.split('*')
            date = str_list[0].strip()
            air_tem = float(str_list[1])

            date_list.append(date)
            air_list.append(air_tem)
    return date_list, air_list


# read data

date_list, daily_tem = read_daily_tem('daily_mean_air_tem.txt')

# for date, tem in zip(date_list,daily_tem):
#     print(date,tem)
#     # print(daily_tem)

for idx in range(0, len(daily_tem)):
    tem =  daily_tem[idx]
    # remove abnormal value
    if tem > 100 or tem < -100:
        daily_tem[idx] = daily_tem[idx-1]
        print('One abnormal value is replaced by the value of previous day: %f'%tem)


fig = plt.figure(figsize=(10, 4))
#
x_date = range(0,len(date_list))



# # draw original data
plt.plot(x_date,daily_tem,'k+',linewidth=1,markersize=5,label='Daily mean air temperature')

#######
### linear Regression using OLS, abandoned
# Note the difference in argument order,
# this is not correct, a offset of X will leads to significantly different results.
# X = np.array(x_date)
# y = np.array(daily_tem)
# model = sm.OLS(y, X).fit()
# predictions = model.predict(X)  # make the predictions by the model
# # Print out the statistics
# print(model.summary())
# print(max(predictions) - min(predictions))
# plt.plot(x_date,predictions,'-b',linewidth=2,markersize=5,label='Linear Regression')

######################
## fit the seasonal value using Polynomial, abandoned
# degree = 6
# X = [i%365 for i in range(0, len(daily_tem))]
# coef = polyfit(X, y, degree)
# print('Coefficients: %s' % str(coef))
# # create curve
# curve = list()
# for i in range(len(X)):
#     value = coef[-1]
#     for d in range(degree):
#         value += X[i]**(degree-d) * coef[d]
#     curve.append(value)
# # plot curve over original data
# # pyplot.plot(series.values)
# # pyplot.plot(curve, color='red', linewidth=3)
# print(max(curve) - min(curve))
# plt.plot(x_date,curve,'-r',linewidth=2,markersize=5,label='Curve')

######################
## fit the linear trend using Polynomial with degree = 1
degree = 1
X = np.array(x_date)
coef = polyfit(X, daily_tem, degree)
print('Coefficients: %s' % str(coef))
# create curve
poly_trend = list()
for i in range(len(X)):
    value = coef[-1]
    for d in range(degree):
        value += X[i]**(degree-d) * coef[d]
    poly_trend.append(value)
# plot curve over original data
# pyplot.plot(series.values)
# pyplot.plot(curve, color='red', linewidth=3)
print('max and min diff:',poly_trend[-1] - poly_trend[0])
print('Linear trend mean:', np.mean(np.array(poly_trend)))
plt.plot(x_date,poly_trend,'-b',linewidth=2,markersize=5,label='Linear trend')


######################
## fit the seasonal value using sin function
def sin_func(x, a, b,c,d):
    return a * np.sin(b * (x - c)) + d
# X = [i%365 for i in range(0, len(daily_tem))]
X = np.array(x_date)
# suggested by https://roywright.me/2018/05/21/model-temperature/
params, params_covariance = optimize.curve_fit(sin_func, X, daily_tem, p0=[20, 2*np.pi / 365, 0, 50])
print('fit parameters',params)
print('fit params_covariance',params_covariance)
curve = sin_func(X, params[0], params[1],params[2], params[3])
period = (2*np.pi / params[1])
print('best period',period)
print(max(curve) - min(curve))
plt.plot(x_date,curve,'-r',linewidth=2,markersize=5,label='Seasonal fitting')

# plot residuals
resid_no_season = daily_tem - curve
print(np.mean(resid_no_season))
# plt.plot(x_date,resid_no_season,'k+',linewidth=1,markersize=5,label='Residuals air temperature')


#######
### linear Regression using OLS for residuals air temperature
# Note the difference in argument order,
# Note that: a offset of X will leads to significantly different results.
# X = np.array(x_date)
# y = np.array(resid_no_season)
# model = sm.OLS(y, X).fit()
# predictions = model.predict(X)  # make the predictions by the model
# # # Print out the statistics
# print(model.summary())
# print(max(predictions) - min(predictions))
# plt.plot(x_date,predictions,'-b',linewidth=2,markersize=5,label='Linear Regression')



#
plt.legend()
plt.ylim([-25,25])
plt.xlabel('Year')
plt.ylabel('Daily mean air temperature ($^\circ$C)')
#
# # plt.tick_params(axis='both', which='major', labelsize=16)


# date_list
# x_date
x_tick_loc = []
x_labels = []
for loc, date in zip(x_date, date_list):
    if "-01-01" in date:
        # print(loc,date)
        x_tick_loc.append(loc)
        x_labels.append(date.split('-')[0])
# add 2013
x_tick_loc.append(len(date_list))
x_labels.append('2013')
print(x_tick_loc)
print(x_labels)

#
plt.xticks(x_tick_loc)
plt.gca().set_xticklabels(x_labels)
# plt.tick_params(axis='x', which='minor',length=3)
# plt.tick_params(axis='x', which='major',length=5,rotation=90) #labelsize=10
#
# plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
#
# plt.show()
#
# # output = 'fig_' + str(np.random.randint(1, 10000)) + '.png'
output = 'daily_mean_air_tem.png'
plt.savefig(output, bbox_inches="tight",dpi=300)  # dpi=200, ,dpi=300