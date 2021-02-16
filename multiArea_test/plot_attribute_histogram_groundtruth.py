#!/usr/bin/env python
# Filename: plot_attribute_histogram_groundtruth.py 
"""
introduction: plot histogram of many attributes in the ground truth shapefile

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 16 February, 2021
"""

import os,sys
code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)

import datasets.data_figures as data_figures
import basic_src.io_function as io_function

def draw_area_size_histogram(shp, pre_name, tail,bin_min,bin_max,bin_width,ylim):
    # area in ha
    out_fig = pre_name+'_area_%s.jpg'%tail
    out_hist_info = pre_name+'_bins_area_%s.txt'%tail
    data_figures.draw_one_value_hist(shp, 'INarea', out_fig,out_hist_info, bin_min,bin_max,bin_width,ylim)

    save_dir = 'size_area_hist'
    io_function.mkdir(save_dir)
    io_function.movefiletodir(out_fig,save_dir,overwrite=True)
    io_function.movefiletodir(out_hist_info,save_dir,overwrite=True)
    return out_fig, out_hist_info

def main():

    # run this script in ~/Data/Arctic/canada_arctic/ground_truth_info

    ground_truth_shp = ['Willow_River_Thaw_Slumps_post.shp', 'Banks_Island_slumps_post.shp', 'HotWeatherCreek_slumps_post.shp']
    pre_names = ['WR', 'Banks', 'HotWC']
    for shp, pre_name in zip(ground_truth_shp,pre_names):
        draw_area_size_histogram(shp, pre_name, 'GT', 0,47,2,[0,180])   # bin_min,bin_max,bin_width,ylim (area in ha)


    # draw_one_value_hist(polygons_imgAug16_tp,'INarea','area_imgAug16_tp.jpg','bins_area_imgAug16_tp.txt',0,31,2,[0,100])

    pass



if __name__ == '__main__':
    main()
    pass