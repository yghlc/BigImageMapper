#!/usr/bin/env python
# Filename: identify_cells_contain_map_results.py
"""
introduction: to identify cells or grids, likey contain true positives for validation.

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 04 September, 2025
"""

import os,sys
import time
from optparse import OptionParser

import numpy as np

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)

import basic_src.io_function as io_function
import basic_src.basic as basic
import datasets.vector_gpd as vector_gpd

from datetime import datetime

import geopandas as gpd

def read_numpy_from_file(npy_file_list,col_name):
    for npy in npy_file_list:
        if os.path.basename(npy) ==  f'{col_name}.npy':
            return np.load(npy)

    raise IOError(f'{col_name}.npy does not exist')

def find_grid_base_on_s2_results(grid_gpd, annual_count_thr=5, npy_file_list = None):

    # s2 count
    s2_count_columns = ['s2_2018_A', 's2_2019_A', 's2_2020_A', 's2_2021_A','s2_2022_A','s2_2023_A', 's2_2024_A']
    s2_area_columns = ['s2_2018_C', 's2_2019_C', 's2_2020_C', 's2_2021_C','s2_2022_C','s2_2023_C', 's2_2024_C']

    if npy_file_list is not None:
        count_array_list =  [read_numpy_from_file(npy_file_list, item) for item in s2_count_columns ]
        area_array_list =  [read_numpy_from_file(npy_file_list, item) for item in s2_area_columns ]
    else:
        count_array_list = [np.array(grid_gpd[item]) for item in s2_count_columns ]
        area_array_list = [ np.array(grid_gpd[item]) for item in s2_area_columns ]

    count_array_2d = np.vstack(count_array_list)
    area_array_2d = np.vstack(area_array_list)

    print('count_array_2d', count_array_2d.shape)
    print('area_array_2d', area_array_2d.shape)

    # selection based on the count
    # if 1 or more there, count a 1, otherwise, as zero
    count_array_2d_binary = (count_array_2d > 0).astype(int)
    count_array_2d_binary_sum = np.sum(count_array_2d_binary,axis=0)    # sum across different years
    print('count_array_2d_binary_sum', count_array_2d_binary_sum.shape)
    b_select_on_count = count_array_2d_binary_sum  >= annual_count_thr
    print('b_select_on_count:', )

    # selection based on area changes





def test_find_grid_base_on_s2_results():

    npy_file_list = io_function.get_file_list_by_ext('.npy','./', bsub_folder=False)
    find_grid_base_on_s2_results(None, npy_file_list=npy_file_list)


    pass


def identify_cells_contain_true_results(grid_gpd, save_path):

    find_grid_base_on_s2_results(grid_gpd)

    pass

def main(options, args):
    grid_path = args[0]
    save_path = options.save_path

    t0 = time.time()
    grid_gpd = gpd.read_file(grid_path)
    t1 = time.time()
    print(f'Loaded grid vector file, containing {len(grid_gpd)} cells, {len(grid_gpd.columns)} columns, cost {t1-t0} seconds')
    print('column names:', grid_gpd.columns.to_list())

    identify_cells_contain_true_results(grid_gpd, save_path)


    pass


if __name__ == '__main__':

    test_find_grid_base_on_s2_results()
    sys.exit(0)

    usage = "usage: %prog [options] grid_vector "
    parser = OptionParser(usage=usage, version="1.0 2025-9-4")
    parser.description = 'Introduction: identify grid (cells) that likely contain true mapping results '

    parser.add_option("-s", "--save_path",
                      action="store", dest="save_path",default="select_cells.gpkg",
                      help="the save path ")

    parser.add_option("-i", "--input_txt",
                      action="store", dest="input_txt",
                      help="the input txt contain column name and vector path (column_name, vector_path)")

    # parser.add_option("-p", "--process_num",
    #                   action="store", dest="process_num",type=int, default=16,
    #                   help="the process number ")

    # parser.add_option("-b", "--using_bounding_box",
    #                   action="store_true", dest="using_bounding_box",default=False,
    #                   help="whether use the boudning boxes of polygons, this can avoid some invalid"
    #                        " polygons and be consistent with YOLO output")




    (options, args) = parser.parse_args()
    # print(options.no_label_image)
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)