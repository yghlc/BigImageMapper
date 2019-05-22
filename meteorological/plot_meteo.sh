#!/usr/bin/env bash


# plot meteorological data series, including temperature, precipitation, and so on

#authors: Huang Lingcao
#email:huanglingcao@gmail.com
#add time: 19 May, 2019


station_no=52908  # wu dao liang station

#station_no=56004  # a station between wu dao liang and tang gu la

#station_no=52818  # Golmud

#station_no=58843 # a station in Xiapu

## only have data from 2010-2017
#cma_dir=/Users/huanglingcao/Data/meteorological_data/4lingcao
## plot precipitation
#plot_meteo_series.py ${cma_dir}/pre_data -s ${station_no} -d pre
#
## plot ground temperature
#plot_meteo_series.py ${cma_dir}/gst_data -s ${station_no} -d gst
#
## plot air temperature
#plot_meteo_series.py ${cma_dir}/tem_data -s ${station_no} -d tem



#data from 1956-2017
cma_dir="/Users/huanglingcao/Data/meteorological_data/CMA_data/中国地面气候资料日值数据集V3.0 1956-2017/v3.0日数据"
# plot precipitation (there are a few folders, don't know which one)
plot_meteo_series.py "${cma_dir}"/全降水 -s ${station_no} -d pre

# plot ground temperature
#plot_meteo_series.py "${cma_dir}"/gst_data -s ${station_no} -d gst

# plot air temperature
#plot_meteo_series.py "${cma_dir}"/气温tem -s ${station_no} -d tem




