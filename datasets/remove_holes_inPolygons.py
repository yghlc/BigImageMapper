#!/usr/bin/env python
# Filename: remove_holes_inPolygons.py
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 22 April, 2021
"""


import os,sys
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(sys.argv[0]), '..')
sys.path.insert(0, code_dir)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import basic_src.io_function as io_function
import basic_src.basic as basic

import vector_gpd

def main(options, args):
    polygons_shp = args[0]

    output = options.output
    if output is None:
        output = io_function.get_name_by_adding_tail(polygons_shp,'noholes')

    vector_gpd.fill_holes_in_polygons_shp(polygons_shp, output)
    basic.outputlogMessage('saving no hole polygons to %s'%output)

def test_fill_holes_in_polygons_shp():
    dir = os.path.expanduser('~/Data/Arctic/canada_arctic/Willow_River/training_polygons')
    shp_path = os.path.join(dir,'WR_training_polygons_v4.shp')
    output = io_function.get_name_by_adding_tail(shp_path, 'noholes')

    vector_gpd.fill_holes_in_polygons_shp(shp_path, output)
    basic.outputlogMessage('saving no hole polygons to %s'%output)



if __name__ == '__main__':
    usage = "usage: %prog [options] shp_file"
    parser = OptionParser(usage=usage, version="1.0 2019-1-4")
    parser.description = 'Introduction: remove holes in polygons'

    parser.add_option("-o", "--output",
                      action="store", dest="output",  # default='save_polygon.shp',
                      help="save file path")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)