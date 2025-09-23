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
from collections import Counter

def read_numpy_from_file(npy_file_list,col_name):
    for npy in npy_file_list:
        if os.path.basename(npy) ==  f'{col_name}.npy':
            return np.load(npy)

    raise IOError(f'{col_name}.npy does not exist in the list: {npy_file_list}')

def read_count_area_array(grid_gpd,count_columns,area_columns, npy_file_list = None):
    if npy_file_list is not None:
        count_array_list =  [read_numpy_from_file(npy_file_list, item) for item in count_columns ]
        area_array_list =  [read_numpy_from_file(npy_file_list, item) for item in area_columns ]
    else:
        count_array_list = [np.array(grid_gpd[item]) for item in count_columns ]
        area_array_list = [ np.array(grid_gpd[item]) for item in area_columns ]

    count_array_2d = np.vstack(count_array_list)
    area_array_2d = np.vstack(area_array_list)
    return count_array_2d, area_array_2d

def find_grid_base_on_s2_results(grid_gpd, annual_count_thr=5, area_trend_thr=0.01, npy_file_list = None):

    # s2 count
    s2_area_columns = ['s2_2018_A', 's2_2019_A', 's2_2020_A', 's2_2021_A','s2_2022_A','s2_2023_A', 's2_2024_A']
    s2_count_columns = ['s2_2018_C', 's2_2019_C', 's2_2020_C', 's2_2021_C','s2_2022_C','s2_2023_C', 's2_2024_C']

    count_array_2d, area_array_2d = read_count_area_array(grid_gpd, s2_count_columns, s2_area_columns, npy_file_list=npy_file_list)

    print('count_array_2d', count_array_2d.shape)
    print('area_array_2d', area_array_2d.shape)

    # selection based on the count
    # if 1 or more there, count a 1, otherwise, as zero
    count_array_2d_sum = np.sum(count_array_2d,axis=0)    # sum across different years
    count_array_2d_binary = (count_array_2d > 0).astype(int)
    count_array_2d_binary_sum = np.sum(count_array_2d_binary,axis=0)    # sum across different years
    print('count_array_2d_binary_sum', count_array_2d_binary_sum.shape, np.min(count_array_2d_binary_sum),
          np.max(count_array_2d_binary_sum), np.mean(count_array_2d_binary_sum))
    b_select_on_count = count_array_2d_binary_sum  >= annual_count_thr
    print('b_select_on_count:', b_select_on_count.shape, b_select_on_count)

    # selection based on area changes
    area_array_2d_sum = np.sum(area_array_2d, axis=0)  # sum across different years
    x = np.arange(area_array_2d.shape[0])  # shape (7,)
    # Calculate slopes (trend) per column, # trends[i] is the trend (slope) for column i
    trends = np.polyfit(x, area_array_2d, deg=1)[0]  # shape (13304783,)
    print('trends', trends.shape, trends)
    b_select_on_area_trend = trends >= area_trend_thr
    print('b_select_on_area_trend:', b_select_on_area_trend.shape, b_select_on_area_trend)

    # conduct the select
    select_idx = np.logical_and(b_select_on_count, b_select_on_area_trend)
    if grid_gpd is None:
        print('For testing, grid_gpd is None')
        return None
    # grid_gpd_sel = grid_gpd[select_idx]
    # grid_gpd_sel['s2_occur'] = count_array_2d_binary_sum[select_idx]
    # grid_gpd_sel['s2_area_trend'] = trends[select_idx]
    s2_occur = count_array_2d_binary_sum[select_idx]
    s2_area_trend = trends[select_idx]
    s2_count_sum = count_array_2d_sum[select_idx]
    s2_area_sum = area_array_2d_sum[select_idx]
    basic.outputlogMessage(f'Select {len(s2_occur)} cells from {len(select_idx)} based on S2 mapping results')
    return select_idx, s2_occur, s2_area_trend, s2_count_sum, s2_area_sum


def find_grid_base_on_DEM_results(grid_gpd, npy_file_list = None):

    # DEM
    dem_area_columns = ['samElev_A', 'comImg_A']
    dem_count_columns = ['samElev_C', 'comImg_C']

    count_array_2d, area_array_2d = read_count_area_array(grid_gpd, dem_count_columns, dem_area_columns,
                                                          npy_file_list=npy_file_list)

    print('count_array_2d', count_array_2d.shape, np.min(count_array_2d, axis=1), np.max(count_array_2d,axis=1), np.mean(count_array_2d,axis=1) )
    print('area_array_2d', area_array_2d.shape, np.min(area_array_2d,axis=1), np.max(area_array_2d,axis=1), np.mean(area_array_2d,axis=1))




def test_find_grid_base_on_s2_results():
    npy_file_list = io_function.get_file_list_by_ext('.npy','./', bsub_folder=False)
    find_grid_base_on_s2_results(None, npy_file_list=npy_file_list)

def test_find_grid_base_on_DEM_results():
    npy_file_list = io_function.get_file_list_by_ext('.npy', './', bsub_folder=False)
    find_grid_base_on_DEM_results(None, npy_file_list=npy_file_list)


def load_training_data_from_validate_jsons(validate_json_list,save_path="valid_res_dict.json"):
    valid_res_dict = {}
    for v_file in validate_json_list:
        data_dict = io_function.read_dict_from_txt_json(v_file)
        res_list = []
        h3_id = data_dict['h3ID']
        for key in data_dict.keys():
            if key == "h3ID":
                continue
            res_list.append(data_dict[key]['ValidateResult'])

        if len(res_list) == 1:
            valid_res_dict[h3_id] = res_list[0]
        elif len(res_list) == 2:
            if len(set(res_list)) != 1:
                basic.outputlogMessage(f'Warning, the validation results in {v_file} from two contributors disagree, choose the first one')
            valid_res_dict[h3_id] = res_list[0]
        elif len(res_list) > 2:
            if len(set(res_list)) != 1:
                basic.outputlogMessage(f'Warning, the validation results in {v_file} from {len(res_list)} contributors disagree, '
                                       f'choose the common')
            # Count the occurrences of each element
            counts = Counter(res_list)
            value, frequency = counts.most_common(1)[0]
            valid_res_dict[h3_id] = value
        else:
            raise ValueError(f'No validation results in {v_file}')

    io_function.save_dict_to_txt_json(save_path,valid_res_dict)

    return valid_res_dict

def test_load_training_data_from_validate_jsons():
    data_dir = os.path.expanduser('~/Data/rts_ArcticDEM_mapping/validation/select_by_s2_result_png')
    json_list = io_function.get_file_list_by_pattern(data_dir,'*/validated*.json')
    load_training_data_from_validate_jsons(json_list)
    pass

def auto_find_positive_grids(grid_gpd,validate_json_list):
    # using machine learning algorithm to find grid that likely contains thaw targets






    pass

def identify_cells_contain_true_results(grid_gpd, save_path):

    # select based on sentinel-2
    select_idx, s2_occur, s2_area_trend, s2_count_sum, s2_area_sum =\
        find_grid_base_on_s2_results(grid_gpd)
    select_grid_gpd = grid_gpd[select_idx]
    select_grid_gpd['s2_occur'] = s2_occur
    select_grid_gpd['s2_area_trend'] = s2_area_trend
    select_grid_gpd['s2_count_sum'] = s2_count_sum
    select_grid_gpd['s2_area_sum'] = s2_area_sum

    select_grid_gpd.to_file(save_path)

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

    # test_find_grid_base_on_s2_results()
    # test_find_grid_base_on_DEM_results()
    test_load_training_data_from_validate_jsons()
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