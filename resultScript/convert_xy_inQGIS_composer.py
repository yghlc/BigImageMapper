#!/usr/bin/env python
# Filename: convert_xy_inQGIS_composer.py
"""
introduction: Convert the map composer extent in QGIS

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 10 February, 2019
"""

import os, sys
HOME = os.path.expanduser('~')
# path of DeeplabforRS
codes_dir2 =  HOME +'/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)

import basic_src.map_projection as map_projection

# copy the extent in map composer, then convert it
# x=[10342265.279,10343901.383]
# y=[4142825.520,4144151.829]


# x=[10342089.718,10344220.996]
# y=[4142571.700,4144141.096]

# x=[8417260.191,11652193.795]
# y=[2854840.549,4908864.529]

# x=[10296000.000,10410000.000]
# y=[4119000.000,4189000.117]

# x=[10313018.123,10315612.801]
# y=[4137593.479,4139631.897]

# x=[10305255.371,10307836.655]
# y=[4158766.200,4160764.500]

x=[10316815.729,10322234.117]
y=[4174269.953,4177294.862]


# convert to UTM 46N
map_projection.convert_points_coordinate_epsg(x,y,3857,32646)
print(x,y)
