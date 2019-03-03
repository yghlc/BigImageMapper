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

def draw_two_list_scatter(x_list,y_list,output,xlabel,ylabel,text_loc_detX,text_locY,color='grey',hatch=""):
    """
    draw a scatter of two attributes

    Returns:True if successful, False otherwise

    """
    x_values = x_list
    y_values = y_list

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

    plt.scatter(x_values, y_values,marker='x',color=color) #marker='^'

    plt.gcf().subplots_adjust(left=0.15)
    plt.gcf().subplots_adjust(bottom=0.15)   # reserve space for label
    # plt.xlabel(attribute,fontsize=15)
    # # plt.ylabel("Frequency")
    # plt.ylabel("Number",fontsize=15)  #
    # # plt.title('Histogram of '+attribute)
    # # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    # # plt.axis([40, 160, 0, 0.03])

    ax.tick_params(labelsize=20)
    plt.xlabel(xlabel,fontsize=20)
    plt.ylabel(ylabel,fontsize=20)

    # # marked  values
    iou_thresholds = [0.5]
    for iou_thr in iou_thresholds:
        ax.axvline(x=iou_thr,color='k',linewidth=0.8,linestyle='--')
        # ax.text(area+100, 0.55, '%d $\mathrm{m^2}$'%area, rotation=90,fontsize=20)
        ax.text(iou_thr+text_loc_detX, text_locY, '%.1f ' % iou_thr, rotation=90, fontsize=20)


    # plt.grid(True)
    plt.savefig(output)
    basic.outputlogMessage("Output figures to %s"%os.path.abspath(output))

def draw_two_attribute_scatter(shp_file,field1,field2,output,logfile):

    x_values = read_attribute(shp_file, field1)
    y_values = read_attribute(shp_file, field2)
    if field1 == 'INarea':                      # m^2 to ha
        x_values = [item/10000.0 for item in x_values]
    if field2 == 'INarea':                      # m^2 to ha
        y_values = [item/10000.0 for item in y_values]

    xlabel = 'IOU value'
    ylabel = 'null'
    text_loc_detX=0.02
    text_loc_Y = 20
    if field2 =='INarea':
        text_loc_Y = 20
        ylabel = 'Area ($ha$)'
    elif field2 =='adj_count':
        text_loc_Y = 5
        ylabel = 'Count'
    elif field2 == 'INperimete':
        text_loc_Y = 4000
        ylabel = 'Perimeter ($m$)'
    elif field2 == 'circularit':
        text_loc_Y = 0.4
        ylabel = 'Circularity'


    draw_two_list_scatter(x_values,y_values,output,xlabel,ylabel,text_loc_detX,text_loc_Y)

    io_function.move_file_to_dst('processLog.txt', os.path.join(out_dir, logfile), overwrite=True)
    io_function.move_file_to_dst(output, os.path.join(out_dir, output), overwrite=True)


out_dir=HOME+'/Data/Qinghai-Tibet/beiluhe/result/result_paper_mapping_RTS_dl_beiluhe'

ground_truth = HOME + '/Data/Qinghai-Tibet/beiluhe/result/result_paper_mapping_RTS_dl_beiluhe/' \
                         'identified_ThawSlumps_prj_post.shp'

result_imgAug16 = HOME + '/Data/Qinghai-Tibet/beiluhe/result/result_paper_mapping_RTS_dl_beiluhe/' \
                         'img_aug_test_results/BLH_basin_deeplabV3+_1_exp9_iter30000_post_imgAug16.shp'

polygons_imgAug16_tp = HOME + '/Data/Qinghai-Tibet/beiluhe/result/result_paper_mapping_RTS_dl_beiluhe/' \
                         'img_aug_test_results/BLH_basin_deeplabV3+_1_exp9_iter30000_post_imgAug16_TP.shp'

# polyon without post-processing (removing polygons based on their areas)
shp_imgAug16_NOpost_tp=os.path.join(out_dir,'img_aug_test_results/BLH_basin_deeplabV3+_1_exp9_iter30000_imgAug16_TP.shp')
shp_imgAug17_NOpost_tp=os.path.join(out_dir,'img_aug_test_results/BLH_basin_deeplabV3+_1_exp9_iter30000_imgAug17_TP.shp')

draw_two_attribute_scatter(result_imgAug16,'IoU','INarea','iou_area_imgAug16_scatter.jpg','bins_iou_area_imgAug16.txt')

draw_two_attribute_scatter(result_imgAug16,'IoU','INperimete','iou_peri_imgAug16_scatter.jpg','bins_iou_peri_imgAug16.txt')

draw_two_attribute_scatter(result_imgAug16,'IoU','circularit','iou_circ_imgAug16_scatter.jpg','bins_iou_circ_imgAug16.txt')

draw_two_attribute_scatter(shp_imgAug16_NOpost_tp,'IoU','INarea','iou_area_imgAug16_NOpost_tp_scatter.jpg','bins_iou_area_imgAug16_NOpost_tp.txt')
draw_two_attribute_scatter(shp_imgAug17_NOpost_tp,'IoU','INarea','iou_area_imgAug17_NOpost_tp_scatter.jpg','bins_iou_area_imgAug17_NOpost_tp.txt')

# intersection of ground truth and polyons without post-processing
intersect_ground_truth_imgAug17_NOpost_tp=os.path.join(out_dir,'intersect_ground_truth_imgAug17_NOpost_tp.shp')
intersect_ground_truth_imgAug16_NOpost_tp=os.path.join(out_dir,'intersect_ground_truth_imgAug16_NOpost_tp.shp')

draw_two_attribute_scatter(intersect_ground_truth_imgAug16_NOpost_tp,'IoU','adj_count',
                           'iou_count_intersect_imgAug16_NOpost_tp_scatter.jpg','bins_iou_count_intersect_imgAug16_NOpost_tp.txt')
draw_two_attribute_scatter(intersect_ground_truth_imgAug17_NOpost_tp,'IoU','adj_count',
                           'iou_count_intersect_imgAug17_NOpost_tp_scatter.jpg','bins_iou_count_intersect_imgAug17_NOpost_tp.txt')