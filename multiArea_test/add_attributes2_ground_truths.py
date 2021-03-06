#!/usr/bin/env python
# Filename: add_attributes2_ground_truths.py 
"""
introduction: add attributes to multiple ground truth polygons

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 16 February, 2021
"""

import os, sys
code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
sys.path.insert(0, os.path.join(code_dir,'datasets'))           # for some modules in this folders
import parameters
import basic_src.io_function as io_function
import basic_src.basic as basic

# import workflow.postProcess as postProcess
from datasets.get_polygon_attributes import add_polygon_attributes


def main():
    para_file = 'main_para.ini'
    io_function.is_file_exist(para_file)
    area_ini_list =['area_Willow_River.ini', 'area_Banks_east_nirGB.ini', 'area_Ellesmere_Island_nirGB.ini']
    for area_ini in area_ini_list:
        io_function.is_file_exist(area_ini)
        ground_truth_shp = parameters.get_file_path_parameters(area_ini, 'validation_shape')

        # save to current folder
        save_info_shp = io_function.get_name_by_adding_tail(ground_truth_shp,'post')
        save_info_shp = os.path.join(os.getcwd(), os.path.basename(save_info_shp))
        if os.path.isfile(save_info_shp):
            basic.outputlogMessage('%s already exist, skip %s'%(save_info_shp, area_ini))
            continue
        add_polygon_attributes(ground_truth_shp,save_info_shp,para_file,area_ini)


if __name__ == '__main__':
    main()
    pass





