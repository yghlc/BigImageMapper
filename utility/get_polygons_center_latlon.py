#!/usr/bin/env python
# Filename: get_polygons_center_latlon.py 
"""
introduction: calculate the geometric centers of polygons and get their latitude and longitude.

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 25 February, 2022
"""

import os, sys
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)

import basic_src.io_function as io_function
import basic_src.basic as basic
import datasets.vector_gpd as vector_gpd
import basic_src.map_projection as map_projection


def calculate_center_latlon(input_vector, save_path,b_save2shp=False):
    io_function.is_file_exist(input_vector)
    epsg_info = map_projection.get_raster_or_vector_srs_info(input_vector, 'epsg')
    epsg_int = int(epsg_info.split(':')[1])

    polygons = vector_gpd.read_polygons_gpd(input_vector,b_fix_invalid_polygon=False)
    poly_center = [vector_gpd.get_polygon_centroid(item) for item in polygons ]

    # to list for the input
    x = [item.x for item in poly_center]
    y = [item.y for item in poly_center]
    # in-place change in x, and y
    # print(x[0],y[0])
    # print(os.getenv('PATH'))
    if map_projection.convert_points_coordinate_epsg(x, y, epsg_int, 4326):  # to 'EPSG:4326'
        # print(x[0], y[0])
        pass
    else:
        raise ValueError('error in convert coordinates')

    # save to file
    save_lines = [ '%s,%s\n'%(str(xx),str(yy)) for xx, yy in zip(x,y) ]
    with open(save_path,'w') as f_obj:
        f_obj.writelines(save_lines)
        basic.outputlogMessage('saved latitude and longitude of polygons to %s'%save_path)

    ext_save_path = io_function.get_name_by_adding_tail(save_path,'ext')
    delta = map_projection.meters_to_degrees_onEarth(1500)  #calculate distance in degree
    with open(ext_save_path,'w') as f_obj:
        for xx, yy in zip(x, y):
            left_x = xx - delta
            right_x = xx + delta
            up_yy = yy + delta
            down_yy = yy - delta
            f_obj.writelines('%f,%f,%f,%f\n'%(left_x,down_yy,right_x,up_yy))

    # write the value to shapefile
    attributes = {'centerLat':y, 'centerLon':x}
    if b_save2shp:
        vector_gpd.add_attributes_to_shp(input_vector,attributes)
        basic.outputlogMessage('saved polygons latitude and longitude to %s' % input_vector)


def main(options, args):
    shp_path = args[0]
    b_save2shp = options.save_latlon_orgFile

    output_path = options.output_latlon
    if output_path is None:
        output_path = os.path.splitext(os.path.basename(shp_path))[0] + '_latlon.txt'

    calculate_center_latlon(shp_path,output_path,b_save2shp=b_save2shp)

def test_calculate_center_latlon():
    shp_path = os.path.expanduser('~/Data/Arctic/alaska/autoMapping/alaskaNS_yolov4_5/result_backup/'
                                  'alaNS_hillshadeHWline_2008to2017_alaskaNS_yolov4_5_exp5_1/alaNS_hillshadeHWline_2008to2017_alaskaNS_yolov4_5_exp5_post_1_select_manuSelect.shp')

    b_save2shp = True
    output_path = os.path.splitext(os.path.basename(shp_path))[0] + '_latlon.txt'
    calculate_center_latlon(shp_path, output_path, b_save2shp=b_save2shp)


if __name__ == "__main__":
    # test_calculate_center_latlon()
    usage = "usage: %prog [options] polygons_paths"
    parser = OptionParser(usage=usage, version="1.0 2018-3-22")
    parser.description = 'Introduction: calculate the geometric centers of polygons and their latitude and longitude '

    parser.add_option("-s", "--save_latlon_orgFile",
                      action="store_true", dest="save_latlon_orgFile", default=False,
                      help="set this, will save the lattitude and longitude to the original vector file")

    parser.add_option("-o", "--output_latlon",
                      action="store", dest="output_latlon",
                      help="save path for latitude and longitude in txt file")


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)


    main(options, args)