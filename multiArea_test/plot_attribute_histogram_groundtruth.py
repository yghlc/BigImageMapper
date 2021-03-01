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

def move_files(save_dir, out_fig, out_hist_info):
    if os.path.isdir(save_dir) is False:
        io_function.mkdir(save_dir)
    trim_fig = io_function.get_name_by_adding_tail(out_fig,'trim')
    os.system('convert -trim %s %s'%(out_fig, trim_fig))
    io_function.movefiletodir(trim_fig,save_dir,overwrite=True)
    io_function.delete_file_or_dir(out_fig)
    # io_function.movefiletodir(out_fig,save_dir,overwrite=True)
    io_function.movefiletodir(out_hist_info,save_dir,overwrite=True)

def draw_area_size_histogram(shp, pre_name, tail,bin_min,bin_max,bin_width,ylim):
    # area in ha
    out_fig = pre_name+'_area_%s.jpg'%tail
    out_hist_info = pre_name+'_bins_area_%s.txt'%tail
    data_figures.draw_one_value_hist(shp, 'INarea', out_fig,out_hist_info, bin_min,bin_max,bin_width,ylim)

    save_dir = 'size_area_hist'
    move_files(save_dir,out_fig, out_hist_info)


def draw_dem_attributes_histogram(shp, pre_name, tail,bin_min,bin_max,bin_width,ylim):
    save_dir = 'dem_hist'
    # min elvetion
    out_fig = pre_name+'_dem_min_%s.jpg'%tail
    out_hist_info = pre_name+'_bins_dem_min_%s.txt'%tail
    data_figures.draw_one_value_hist(shp, 'dem_min', out_fig,out_hist_info, bin_min,bin_max,bin_width,ylim)
    move_files(save_dir, out_fig, out_hist_info)

    # mean elevation
    out_fig = pre_name+'_dem_mean_%s.jpg'%tail
    out_hist_info = pre_name+'_bins_dem_mean_%s.txt'%tail
    data_figures.draw_one_value_hist(shp, 'dem_mean', out_fig,out_hist_info, bin_min,bin_max,bin_width,ylim)
    move_files(save_dir, out_fig, out_hist_info)

    # max elevation
    out_fig = pre_name+'_dem_max_%s.jpg'%tail
    out_hist_info = pre_name+'_bins_dem_max_%s.txt'%tail
    data_figures.draw_one_value_hist(shp, 'dem_max', out_fig,out_hist_info, bin_min,bin_max,bin_width,ylim)
    move_files(save_dir, out_fig, out_hist_info)

def draw_slope_attributes_histogram(shp, pre_name, tail,bin_min,bin_max,bin_width,ylim):
    save_dir = 'slope_hist'
    # min elvetion
    out_fig = pre_name+'_slo_min_%s.jpg'%tail
    out_hist_info = pre_name+'_bins_slo_min_%s.txt'%tail
    data_figures.draw_one_value_hist(shp, 'slo_min', out_fig,out_hist_info, bin_min,bin_max,bin_width,ylim)
    move_files(save_dir, out_fig, out_hist_info)

    # mean elevation
    out_fig = pre_name+'_slo_mean_%s.jpg'%tail
    out_hist_info = pre_name+'_bins_slo_mean_%s.txt'%tail
    data_figures.draw_one_value_hist(shp, 'slo_mean', out_fig,out_hist_info, bin_min,bin_max,bin_width,ylim)
    move_files(save_dir, out_fig, out_hist_info)

    # max elevation
    out_fig = pre_name+'_slo_max_%s.jpg'%tail
    out_hist_info = pre_name+'_bins_slo_max_%s.txt'%tail
    data_figures.draw_one_value_hist(shp, 'slo_max', out_fig,out_hist_info, bin_min,bin_max,bin_width,ylim)
    move_files(save_dir, out_fig, out_hist_info)

def draw_dem_diff_attributes_histogram(shp, pre_name, tail,bin_min,bin_max,bin_width,ylim):
    save_dir = 'dem_diff_hist'
    # min elvetion
    out_fig = pre_name+'_demD_min_%s.jpg'%tail
    out_hist_info = pre_name+'_bins_demD_min_%s.txt'%tail
    data_figures.draw_one_value_hist(shp, 'demD_min', out_fig,out_hist_info, bin_min,bin_max,bin_width,ylim)
    move_files(save_dir, out_fig, out_hist_info)

    # mean elevation
    out_fig = pre_name+'_demD_mean_%s.jpg'%tail
    out_hist_info = pre_name+'_bins_demD_mean_%s.txt'%tail
    data_figures.draw_one_value_hist(shp, 'demD_mean', out_fig,out_hist_info, bin_min,bin_max,bin_width,ylim)
    move_files(save_dir, out_fig, out_hist_info)

    # max elevation
    out_fig = pre_name+'_demD_max_%s.jpg'%tail
    out_hist_info = pre_name+'_bins_demD_max_%s.txt'%tail
    data_figures.draw_one_value_hist(shp, 'demD_max', out_fig,out_hist_info, bin_min,bin_max,bin_width,ylim)
    move_files(save_dir, out_fig, out_hist_info)

def main():

    # run this script in ~/Data/Arctic/canada_arctic/ground_truth_info

    ground_truth_shp = ['Willow_River_Thaw_Slumps_post.shp', 'Banks_Island_slumps_post.shp', 'HotWeatherCreek_slumps_post.shp']
    pre_names = ['WR', 'Banks', 'HotWC']
    for shp, pre_name in zip(ground_truth_shp,pre_names):
        draw_area_size_histogram(shp, pre_name, 'GT', 0,47,2,[0,180])   # bin_min,bin_max,bin_width,ylim (area in ha)

        draw_dem_attributes_histogram(shp, pre_name, 'GT', -100,1000,50,[0,90])

        draw_slope_attributes_histogram(shp, pre_name, 'GT', 0, 77, 3, [0,260])

        draw_dem_diff_attributes_histogram(shp, pre_name, 'GT', -30, 21, 2, [0,160])


    pass



if __name__ == '__main__':
    main()
    pass