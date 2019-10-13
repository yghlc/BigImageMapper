#!/usr/bin/env python
# Filename: remove_falsePositives 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 11 October, 2019
"""

import os,sys
from optparse import OptionParser

HOME = os.path.expanduser('~')
# path of DeeplabforRS
codes_dir2 = HOME + '/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)


import basic_src.io_function as io_function
import basic_src.basic as basic
import vector_features
from vector_features import shape_opeation

def remove_polygons(shapefile,field_name, threshold, bsmaller,output):
    #  remove the not narrow polygon based on ratio_p_a
    operation_obj = shape_opeation()
    if operation_obj.remove_shape_baseon_field_value(shapefile, output, field_name, threshold, smaller=bsmaller) is False:
        return False


def main(options, args):
    polygons_shp = args[0]

    output = options.output

    assert io_function.is_file_exist(polygons_shp)

    # remove polygons based on area
    rm_area_save_shp = io_function.get_name_by_adding_tail(polygons_shp,'rmArea')
    area_thr = 1000  #10 pixels
    b_smaller = True
    remove_polygons(polygons_shp, 'INarea', area_thr, b_smaller, rm_area_save_shp)

    # remove  polygons based on slope information
    rm_slope_save_shp1 = io_function.get_name_by_adding_tail(polygons_shp, 'rmslope1')
    slope_small_thr = 2
    b_smaller = True
    remove_polygons(rm_area_save_shp, 'slo_mean', slope_small_thr, b_smaller, rm_slope_save_shp1)


    rm_slope_save_shp2 = io_function.get_name_by_adding_tail(polygons_shp, 'rmslope2')
    slope_large_thr = 20
    b_smaller = False
    remove_polygons(rm_slope_save_shp1, 'slo_mean', slope_large_thr, b_smaller, rm_slope_save_shp2)

    # remove polgyons based on dem
    rm_dem_save_shp = output  # final output
    dem_small_thr = 3000
    b_smaller = True
    remove_polygons(rm_slope_save_shp1, 'dem_mean', dem_small_thr, b_smaller, rm_dem_save_shp)



    pass



if __name__ == "__main__":
    usage = "usage: %prog [options] shp_file"
    parser = OptionParser(usage=usage, version="1.0 2019-1-4")
    parser.description = 'Introduction: remove polygons based on an attributes values'

    parser.add_option("-o", "--output",
                      action="store", dest="output",default='save_polygon.shp',
                      help="save file path")


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)