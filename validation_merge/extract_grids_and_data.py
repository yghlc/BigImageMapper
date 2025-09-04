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

import numpy as np

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)

import basic_src.io_function as io_function
import basic_src.basic as basic

def main(options, args):
    grid_path = args[0]
    out_dir = options.out_dir



    pass


if __name__ == '__main__':
    usage = "usage: %prog [options] grid_vector "
    parser = OptionParser(usage=usage, version="1.0 2025-9-4")
    parser.description = 'Introduction: extract multiple images and vector for each grid (cell) for validation '

    parser.add_option("-o", "--out_dir",
                      action="store", dest="out_dir",default="validation_data",
                      help="the output directory ")

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
