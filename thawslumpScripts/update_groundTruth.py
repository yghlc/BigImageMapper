#!/usr/bin/env python
# Filename: update_groundTruth 
"""
introduction: update ground truth polygons from new manual identification

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 25 July, 2019
"""

import os,sys

HOME = os.path.expanduser('~')
# path of DeeplabforRS
codes_dir2 = HOME +'/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)

codes_dir = HOME + '/codes/PycharmProjects/Landuse_DL'
sys.path.insert(0, codes_dir)

import vector_features
from vector_features import shape_opeation

import resultScript.add_info2Pylygons as add_info2Pylygons


org_gtPolygons = HOME + '/Data/Qinghai-Tibet/beiluhe/thaw_slumps/' \
                        'train_polygons_for_planet_2018_luo_Oct23/identified_ThawSlumps_prj.shp'

new_polygons = HOME + '/Data/Qinghai-Tibet/beiluhe/thaw_slumps/' \
                      'thawslump_polygons_2017_438_Luo/2017-10_prj.shp'

# calculate the IoU values and save to "new_polygons"

# org_gtPolygons like ground truths
# new_polygons as mapping results
add_info2Pylygons.add_IoU_values(new_polygons,org_gtPolygons,'IOU_old')

# then check polygons one by one manually, and then add some of them to ground truths









