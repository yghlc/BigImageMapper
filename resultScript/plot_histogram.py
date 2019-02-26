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

import plot_results
import basic_src.io_function as io_function
import basic_src.basic as basic

import rasterio
import numpy as np

import matplotlib
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.axislines import Subplot

def draw_one_attribute_histogram(shp_file,field_name,attribute, output,color='grey',hatch=""):
    """
    draw the figure of one attribute's histograms
    Args:
        shp_file:  shape file path
        attribute_name: name of attribute
        output: output the figure

    Returns: True if successful, False otherwise

    """
    values = plot_results.read_attribute(shp_file,field_name)
    if field_name == 'INarea':                      # m^2 to km^2
        values = [item/1000000.0 for item in values]

    fig_obj = plt.figure()  # create a new figure

    ax = Subplot(fig_obj, 111)
    fig_obj.add_subplot(ax)

    # n, bins, patches = plt.hist(values, bins="auto", alpha=0.75,ec="black")  # ec means edge color
    n, bins, patches = ax.hist(values, bins="auto", alpha=0.75, ec="black",linewidth='1.5',color=color,hatch=hatch)
    # print(n,bins,patches)
    # n_label = [str(i) for i in n]
    # plt.hist(values, bins="auto", alpha=0.75, ec="black",label=n_label)

    # plt.gcf().subplots_adjust(bottom=0.15)   # reserve space for label
    # plt.xlabel(attribute,fontsize=15)
    # # plt.ylabel("Frequency")
    # plt.ylabel("Number",fontsize=15)  #
    # plt.title('Histogram of '+attribute)
    # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    # plt.axis([40, 160, 0, 0.03])


    # hide the right and top boxed axis
    ax.axis["right"].set_visible(False)
    ax.axis["top"].set_visible(False)


    # plt.grid(True)
    plt.savefig(output)
    basic.outputlogMessage("Output figures to %s"%os.path.abspath(output))
    basic.outputlogMessage("ncount: " + str(n))
    basic.outputlogMessage("bins: "+ str(bins))
    # plt.show()


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

    print(global_bin_size)
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

def draw_two_list_histogram(shp_file,field_name,ano_list,output,bins=None,color=None,hatch=""):
    """

    Args:
        shp_file:  shape file path
        attribute_name: name of attribute
        output: output the figure

    Returns: True if successful, False otherwise

    """
    values = plot_results.read_attribute(shp_file,field_name)

    x_multi = [values,ano_list]
    fig_obj = plt.figure(figsize=(8,6))  # create a new figure

    ax = Subplot(fig_obj, 111)
    fig_obj.add_subplot(ax)

    # density=True,
    # bins = "auto"
    # n_bins = 10
    # bins = np.arange(4400,5000,50)
    n, bins, patches = ax.hist(x_multi,bins=bins,normed = True, alpha=0.75, ec="black",linewidth='1.5',
                               color=color,hatch=hatch,label=['RTS','Landscape'],rwidth=1)

    # n, bins, patches = ax.hist(values,bins=bins,normed = True, alpha=0.75, ec="black",linewidth='1.5',
    #                            color=['grey'],hatch=hatch,label=['RTS'],stacked=True)

    ax.legend(prop={'size': 12})

    # Create the formatter using the function to_percent. This multiplies all the
    # default labels by 100, making them all percentages
    formatter = FuncFormatter(to_percent)

    # Set the formatter
    plt.gca().yaxis.set_major_formatter(formatter)



    # hide the right and top boxed axis
    ax.axis["right"].set_visible(False)
    ax.axis["top"].set_visible(False)

    # set tick, not work
    # plt.xticks([])
    # plt.yticks([])
    # plt.tick_params(direction='out', length=6, width=2)
    # ax.tick_params(axis='both',direction='out', colors='red',length=0.1)
    # ax.tick_params(axis='y',direction='inout', colors='red', length=10) #,width=50



    # plt.grid(True)
    plt.savefig(output)
    basic.outputlogMessage("Output figures to %s"%os.path.abspath(output))
    basic.outputlogMessage("ncount: " + str(n))
    basic.outputlogMessage("bins: "+ str(bins))
    # plt.show()

def draw_two_values_hist(shp_file,field_name,raster_file,output,logfile,bin_min,bin_max,bin_width):

    raster_values = read_oneband_image_to_1dArray(raster_file)
    bins = np.arange(bin_min, bin_max, bin_width)

    # update
    global global_bin_size
    global_bin_size = bin_width

    draw_two_list_histogram(shp_file, field_name, raster_values, output, bins=bins,color=['black', 'silver'])
    io_function.copy_file_to_dst('processLog.txt', os.path.join(out_dir,logfile), overwrite=True)
    io_function.copy_file_to_dst(output, os.path.join(out_dir,output), overwrite=True)


out_dir=HOME+'/Data/Qinghai-Tibet/beiluhe/result/result_paper_mapping_RTS_dl_beiluhe'

# plot histogram of IOU values.
result_NOimgAug = HOME + '/Data/Qinghai-Tibet/beiluhe/result/result_paper_mapping_RTS_dl_beiluhe/' \
                         'BLH_basin_deeplabV3+_1_exp9_iter30000_post_1.shp'
result_imgAug16 = HOME + '/Data/Qinghai-Tibet/beiluhe/result/result_paper_mapping_RTS_dl_beiluhe/' \
                         'img_aug_test_results/BLH_basin_deeplabV3+_1_exp9_iter30000_post_imgAug16.shp'

ground_truth = HOME + '/Data/Qinghai-Tibet/beiluhe/result/result_paper_mapping_RTS_dl_beiluhe/' \
                         'identified_ThawSlumps_prj_post.shp'

dem=HOME+'/Data/Qinghai-Tibet/beiluhe/DEM/srtm_30/beiluhe_srtm30_utm_basinExt.tif'


# plot histogram of IOU values.
# plot_results.draw_one_attribute_histogram(result_imgAug16, "IoU", "IoU (0-1)", "IoU_imgAug16.jpg")  # ,hatch='-'
# plot_results.draw_one_attribute_histogram(result_NOimgAug, "IoU", "IoU (0-1)", "IoU_NOimgAug.jpg")

# io_function.copy_file_to_dst('processLog.txt',out_dir+'/bins_iou.txt',overwrite=True)
# io_function.copy_file_to_dst('IoU_imgAug16.jpg',out_dir+'/IoU_imgAug16.jpg',overwrite=True)
# io_function.copy_file_to_dst('IoU_NOimgAug.jpg',out_dir+'/IoU_NOimgAug.jpg',overwrite=True)


# plot histogram of PISR values
# plot_results.draw_one_attribute_histogram(ground_truth, "pisr_mean", "pisr", "PISR_ground_truth.jpg")  # ,hatch='-'
# io_function.copy_file_to_dst('processLog.txt',out_dir+'/bins_pisr_gt.txt',overwrite=True)
# io_function.copy_file_to_dst('PISR_ground_truth.jpg',out_dir+'/PISR_ground_truth.jpg',overwrite=True)

# plot histogram of dem values
# dem_values = read_oneband_image_to_1dArray(dem)  # Computed Min/Max=4415.000,5400.000
# # plot_results.draw_one_attribute_histogram(ground_truth, "dem_mean", "dem", "dem_ground_truth.jpg")  # ,hatch='-'
# draw_two_list_histogram(ground_truth, "dem_mean",dem_values,"dem_ground_truth.jpg",color=['black','silver'])
# io_function.copy_file_to_dst('processLog.txt',out_dir+'/bins_dem_gt.txt',overwrite=True)
# io_function.copy_file_to_dst('dem_ground_truth.jpg',out_dir+'/dem_ground_truth.jpg',overwrite=True)

# draw_two_values_hist(ground_truth,"dem_mean",dem,"dem_ground_truth.jpg",'bins_dem_gt.txt',4400,5000,50)

# np.random.seed(19680801)
# n_bins = 10
# x = np.random.randn(1000, 3)
# fig, axes = plt.subplots(nrows=1, ncols=1)
#
# # Make a multiple-histogram of data-sets with different length.
# x_multi = [np.random.randn(n) for n in [10000, 5000, 2000]]
# axes.hist(x_multi, n_bins, histtype='bar')
# axes.set_title('different sample sizes')
# axes.tick_params(axis='both',direction='out',length=10)
#
# fig.tight_layout()
# plt.show()



# plot histogram of slope values
# plot_results.draw_one_attribute_histogram(ground_truth, "slo_mean", "slope", "slope_ground_truth.jpg")  # ,hatch='-'
# io_function.copy_file_to_dst('processLog.txt',out_dir+'/bins_slope_gt.txt',overwrite=True)
# io_function.copy_file_to_dst('slope_ground_truth.jpg',out_dir+'/slope_ground_truth.jpg',overwrite=True)


# clear
os.system('rm processLog.txt')
os.system('rm *.jpg')