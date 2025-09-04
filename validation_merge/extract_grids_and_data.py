#!/usr/bin/env python
# Filename: extract_grids_and_data.py 
"""
introduction: extract images, vector for each grid (cell) for validation

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 04 September, 2025
"""

import os,sys
import time
from optparse import OptionParser



code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)

import basic_src.io_function as io_function
import basic_src.basic as basic

import parameters

import numpy as np
import geopandas as gpd



def main(options, args):
    grid_path = args[0]
    out_dir = options.out_dir

    t0 = time.time()
    grid_gpd = gpd.read_file(grid_path)
    t1 = time.time()
    print(f'Loaded grid vector file, containing {len(grid_gpd)} cells, {len(grid_gpd.columns)} columns, cost {t1-t0} seconds')

    mapping_res_ini = options.mapping_res_ini
    in_vectors_colum_dict = {}
    if mapping_res_ini is not None:
        pass
        # tmp_list = io_function.read_list_from_txt(input_txt)
        # for tmp in tmp_list:
        #     col_and_file = [item.strip() for item in tmp.split(',')]
        #     in_vectors_colum_dict[col_and_file[0]] = col_and_file[1]
    else:
        print('Please set "--mapping_res_ini"')
        return

    if os.path.isdir(out_dir) is False:
        io_function.mkdir(out_dir)





if __name__ == '__main__':
    usage = "usage: %prog [options] grid_vector "
    parser = OptionParser(usage=usage, version="1.0 2025-9-4")
    parser.description = 'Introduction: extract multiple images and vector for each grid (cell) for validation '

    parser.add_option("-o", "--out_dir",
                      action="store", dest="out_dir",default="validation_data",
                      help="the output directory ")

    parser.add_option("-i", "--mapping_res_ini",
                      action="store", dest="mapping_res_ini",
                      help="the ini file contain mapping results and the corresponding rasters")

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
