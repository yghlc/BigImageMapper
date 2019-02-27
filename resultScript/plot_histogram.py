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

def read_oneband_image_to_1dArray(image_path):

    if os.path.isfile(image_path) is False:
        raise IOError("error, file not exist: " + image_path)

    with rasterio.open(image_path) as img_obj:
        # read the all bands (only have one band)
        indexes = img_obj.indexes
        if len(indexes) != 1:
            raise IOError('error, only support one band')

        data = img_obj.read(indexes)
        data_1d = data.flatten()  # convert to one 1d, row first.

        return data_1d

global_bin_size = 50   # remember to change this one
def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.

    # print(global_bin_size)
    # s = str(100 * y*bin_size)
    s = "%.0f"%(100 * y*global_bin_size)

    # The percent symbol needs escaping in latex
    # if matplotlib.rcParams['text.usetex'] is True:
    #     return s + r'$\%$'
    # else:
    #     return s + '%'
    if matplotlib.rcParams['text.usetex'] is True:
        return s
    else:
        return s

def draw_two_list_histogram(shp_file,field_name,ano_list,output,bins=None,labels=None,color=None,hatch=""):
    """

    Args:
        shp_file:  shape file path
        attribute_name: name of attribute
        output: output the figure

    Returns: True if successful, False otherwise

    """
    values = read_attribute(shp_file,field_name)

    x_multi = [values,ano_list]
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(8,6))
    # fig_obj = plt.figure(figsize=(8,6))  # create a new figure
    # ax = Subplot(fig_obj, 111)
    # fig_obj.add_subplot(ax)

    # density=True,
    # bins = "auto"
    # n_bins = 10
    # bins = np.arange(4400,5000,50)
    n, bins, patches = ax.hist(x_multi,bins=bins,density = True, alpha=0.75, ec="black",linewidth='1.5',
                               color=color,hatch=hatch,label=labels,rwidth=1)

    # n, bins, patches = ax.hist(values,bins=bins,normed = True, alpha=0.75, ec="black",linewidth='1.5',
    #                            color=['grey'],hatch=hatch,label=['RTS'],stacked=True)
    fontsize=18
    ax.legend(prop={'size': fontsize})

    # Create the formatter using the function to_percent. This multiplies all the
    # default labels by 100, making them all percentages
    formatter = FuncFormatter(to_percent)

    # Set the formatter
    plt.gca().yaxis.set_major_formatter(formatter)



    # hide the right and top boxed axis
    # ax.axis["right"].set_visible(False)
    # ax.axis["top"].set_visible(False)

    plt.xticks(bins)
    # ax1.get_xaxis().set_ticklabels(layer_num)

    # plt.tick_params(direction='out', length=6, width=2)
    # ax.tick_params(axis='both',direction='out', colors='red',length=0.1)
    ax.tick_params(axis='both',which='both',direction='out', length=7,labelsize=fontsize) #,width=50 #,

    if 'dem' in field_name or 'pisr' in field_name or 'asp' in field_name \
            or 'tpi' in field_name or 'slo' in field_name:
        ax.tick_params(axis='x',labelrotation=90)

    plt.gcf().subplots_adjust(bottom=0.15)
    # plt.grid(True)
    plt.savefig(output)
    basic.outputlogMessage("Output figures to %s"%os.path.abspath(output))
    basic.outputlogMessage("ncount: " + str(n))
    basic.outputlogMessage("bins: "+ str(bins))
    # plt.show()

def draw_one_list_histogram(value_list,output,bins=None,labels=None,color=None,hatch="",xlabelrotation=None):


    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(8,8))
    n, bins, patches = ax.hist(value_list,bins=bins, alpha=0.75, ec="black",linewidth='1.5',
                               color=color,hatch=hatch,label=labels,rwidth=1) #density = True,

    # ax.legend(prop={'size': 12})
    plt.xticks(bins)
    ax.tick_params(axis='both',which='both',direction='out',length=7,labelsize=20) #,width=50 #,
    if xlabelrotation is not None:
        ax.tick_params(axis='x', labelrotation=90)

    # plt.grid(True)
    plt.savefig(output)  #
    basic.outputlogMessage("Output figures to %s"%os.path.abspath(output))
    basic.outputlogMessage("ncount: " + str(n))
    basic.outputlogMessage("bins: "+ str(bins))
    # plt.show()


def draw_two_values_hist(shp_file,field_name,raster_file,output,logfile,bin_min,bin_max,bin_width,labels):

    raster_values = read_oneband_image_to_1dArray(raster_file)
    bins = np.arange(bin_min, bin_max, bin_width)

    # update
    global global_bin_size
    global_bin_size = bin_width

    draw_two_list_histogram(shp_file, field_name, raster_values, output, bins=bins,labels=labels,color=['black', 'silver'])
    io_function.copy_file_to_dst('processLog.txt', os.path.join(out_dir,logfile), overwrite=True)
    io_function.copy_file_to_dst(output, os.path.join(out_dir,output), overwrite=True)


def draw_one_value_hist(shp_file,field_name,output,logfile,bin_min,bin_max,bin_width):

    values = read_attribute(shp_file, field_name)
    if field_name == 'INarea':                      # m^2 to ha
        values = [item/10000.0 for item in values]

    xlabelrotation = None
    if 'area' in field_name or 'INperimete' in field_name or 'circularit' in field_name:
        xlabelrotation = 90

    bins = np.arange(bin_min, bin_max, bin_width)

    # plot histogram of slope values
    # value_list,output,bins=None,labels=None,color=None,hatch=""
    draw_one_list_histogram(values, output,bins=bins,color=['grey'],xlabelrotation=xlabelrotation )  # ,hatch='-'
    io_function.copy_file_to_dst('processLog.txt', os.path.join(out_dir, logfile), overwrite=True)
    io_function.copy_file_to_dst(output, os.path.join(out_dir, output), overwrite=True)


out_dir=HOME+'/Data/Qinghai-Tibet/beiluhe/result/result_paper_mapping_RTS_dl_beiluhe'

# plot histogram of IOU values.
result_NOimgAug = HOME + '/Data/Qinghai-Tibet/beiluhe/result/result_paper_mapping_RTS_dl_beiluhe/' \
                         'BLH_basin_deeplabV3+_1_exp9_iter30000_post_1.shp'
result_imgAug16 = HOME + '/Data/Qinghai-Tibet/beiluhe/result/result_paper_mapping_RTS_dl_beiluhe/' \
                         'img_aug_test_results/BLH_basin_deeplabV3+_1_exp9_iter30000_post_imgAug16.shp'

ground_truth = HOME + '/Data/Qinghai-Tibet/beiluhe/result/result_paper_mapping_RTS_dl_beiluhe/' \
                         'identified_ThawSlumps_prj_post.shp'

dem=HOME+'/Data/Qinghai-Tibet/beiluhe/DEM/srtm_30/beiluhe_srtm30_utm_basinExt.tif'
slope=HOME+'/Data/Qinghai-Tibet/beiluhe/DEM/srtm_30/beiluhe_srtm30_utm_basinExt_slope.tif'
aspect=HOME+'/Data/Qinghai-Tibet/beiluhe/DEM/srtm_30/beiluhe_srtm30_utm_basinExt_apect.tif'

pisr = HOME+'/Data/Qinghai-Tibet/beiluhe/DEM/srtm_30/dem_derived/beiluhe_srtm30_utm_basinExt_PISR_total_perDay.tif'
tpi = HOME+'/Data/Qinghai-Tibet/beiluhe/DEM/srtm_30/dem_derived/beiluhe_srtm30_utm_basinExt_tpi.tif'

####################################################################
# draw one list (attributes)

# iou values
# draw_one_value_hist(result_imgAug16,'IoU','IoU_imgAug16_new.jpg','bins_IoU_imgAug16.txt',0,1.01,0.1)

# iou values
# draw_one_value_hist(result_NOimgAug,'IoU','IoU_NOimgAug_new.jpg','bins_NOimgAug.txt',0,1.01,0.1)

# # area # in ha, min 0.25, max: 29
# draw_one_value_hist(ground_truth,'INarea','area_ground_truth.jpg','bins_area_gt.txt',0,31,2)
#
# # perimeters meters, min 235, max 5898
# draw_one_value_hist(ground_truth,'INperimete','perimeter_ground_truth.jpg','bins_perimeter_gt.txt',200,6300,600)
#
# # circularity 0 - 1
# draw_one_value_hist(ground_truth,'circularit','circularity_ground_truth.jpg','bins_circularity_gt.txt',0,1.01,0.1)


####################################################################
## draw two list together

# dem
# draw_two_values_hist(ground_truth,"dem_mean",dem,"dem_ground_truth.jpg",'bins_dem_gt.txt',4400,5250,50,['RTS','Landscape'])

# slope #Computed Min/Max=0.000,48.435
draw_two_values_hist(ground_truth,"slo_mean",slope,"slope_ground_truth.jpg",'bins_slope_gt.txt',0,20,1,['RTS','Landscape'])

# pisr per day #Computed Min/Max=0.000,9.131
# draw_two_values_hist(ground_truth,"pisr_mean",pisr ,"pisr_ground_truth.jpg",'bins_pisr_gt.txt',8.5,9.15,0.03,['RTS','Landscape'])


# aspect #Computed Min/Max=0.269,360.000, the raster aspect seems not correct
# draw_two_values_hist(ground_truth,"asp_mean",aspect ,"aspect_ground_truth.jpg",'bins_apsect_gt.txt',0,360,15,['RTS','Landscape'])

#TPI # Minimum=-11.919, Maximum=13.788
# draw_two_values_hist(ground_truth,"tpi_mean",tpi ,"tpi_ground_truth.jpg",'bins_tpi_gt.txt',-4,4.1,0.5,['RTS','Landscape'])


# clear
os.system('rm processLog.txt')
os.system('rm *.jpg')