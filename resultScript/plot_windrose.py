#!/usr/bin/env python
# Filename: plot_histogram.py 
"""
introduction: plot histogram

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 23 February, 2019
"""

import os, sys
HOME = os.path.expanduser('~')
# path of DeeplabforRS
codes_dir2 =  HOME +'/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)


import basic_src.io_function as io_function
import basic_src.basic as basic

import rasterio
import numpy as np

import matplotlib
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt


from vector_features import read_attribute

def draw_one_attribute_windrose(shp_file,field_name,attribute, output,color='grey',hatch=""):
    """
    draw the figure of one attribute's wind rose
    Args:
        shp_file:  shape file path
        attribute_name: name of attribute
        output: output the figure

    Returns: True if successful, False otherwise

    """
    values = read_attribute(shp_file,field_name)

    from windrose import WindroseAxes

    wind_dir = np.array(values)
    wind_sd = np.ones(wind_dir.shape[0]) #np.ones(wind_dir.shape[0])  #np.arange(1, wind_dir.shape[0] + 1)
    bins_range = np.arange(1, 2, 1)  # this sets the legend scale

    ax = WindroseAxes.from_ax()
    # ax.bar(wind_dir, wind_sd, normed=True, bins=bins_range,colors=color)
    ax.bar(wind_dir, wind_sd, bins=bins_range, colors=color)  # normed=True, show count, not density
    ax.set_ylim([0,45])
    ax.set_yticks(np.arange(10, 41, step=10))
    ax.set_yticklabels(np.arange(10, 41, step=10),color='red')


    ax.tick_params(labelsize=20)

    # plt.show()
    plt.savefig(output)



    # plt.grid(True)
    # plt.savefig(output)
    # basic.outputlogMessage("Output figures to %s"%os.path.abspath(output))
    # basic.outputlogMessage("ncount: " + str(n))
    # basic.outputlogMessage("bins: "+ str(bins))
    return True



# out_dir=HOME+'/Data/Qinghai-Tibet/beiluhe/result/result_paper_mapping_RTS_dl_beiluhe'
out_dir=HOME+'/Data/Qinghai-Tibet/beiluhe/result/result_paper_mapping_RTS_dl_beiluhe_revised2019'

# ground_truth = HOME + '/Data/Qinghai-Tibet/beiluhe/result/result_paper_mapping_RTS_dl_beiluhe/' \
#                          'identified_ThawSlumps_prj_post.shp'

# aspect=HOME+'/Data/Qinghai-Tibet/beiluhe/DEM/srtm_30/beiluhe_srtm30_utm_basinExt_apect.tif'

# aspect_line=os.path.join(out_dir,'identified_ThawSlumps_MaiinLines_prj.shp')
aspect_line=os.path.join(out_dir,'identified_ThawSlumps_MaiinLines_utm.shp')


# aspect_line_imgAug16_tp=os.path.join(out_dir,'identified_ThawSlumps_MaiinLines_prj_TP.shp')
aspect_line_imgAug22_tp=os.path.join(out_dir,'identified_ThawSlumps_MaiinLines_utm_TP.shp')


# output='aspect_ground_truth_winrose.jpg'
# draw_one_attribute_windrose(ground_truth,'asp_mean','',output ,color='grey',hatch="")
# io_function.copy_file_to_dst(output, os.path.join(out_dir,output), overwrite=True)

# draw wind rose of azimuth from manually draw lines
# output="aspectLine_ground_truth_winrose.jpg"
# draw_one_attribute_windrose(aspect_line, "aspectLine", "Mean Aspect ($^\circ$)", output,color='black')  # ,hatch='/'
# io_function.copy_file_to_dst(output, os.path.join(out_dir,output), overwrite=True)

####### use mapping polygons  ####
# output="aspectLine_imgAug16_tp_winrose.jpg"
# draw_one_attribute_windrose(aspect_line_imgAug16_tp, "aspectLine", "Mean Aspect ($^\circ$)", output,color='black')  # ,hatch='/'
output="aspectLine_imgAug22_tp_winrose.jpg"
draw_one_attribute_windrose(aspect_line_imgAug22_tp, "aspectLine", "Mean Aspect ($^\circ$)", output,color='black')  # ,hatch='/'
io_function.copy_file_to_dst(output, os.path.join(out_dir,output), overwrite=True)
####### use mapping polygons  ####

#
# # clear
# os.system('rm processLog.txt')
os.system('rm *.jpg')