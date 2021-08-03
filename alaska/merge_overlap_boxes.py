#!/usr/bin/env python
# Filename: merge_overlap_boxes.py
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 18 April, 2021
"""
import os,sys
code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters

import datasets.vector_gpd as vector_gpd
import datasets.vector_features as vector_features

import basic_src.io_function as io_function
import basic_src.map_projection as map_projection

import pandas as pd
# import geopandas as gpd

def merge_based_on_adjacent_matrix(in_shp):

    # not working, many box did not merge together.

    process_num = 8
    polygons = vector_gpd.read_polygons_gpd(in_shp)

    print('start building adjacent_matrix')
    adjacent_matrix = vector_gpd.build_adjacent_map_of_polygons(polygons, process_num=process_num)
    print( 'finish building adjacent_matrix')
    if adjacent_matrix is False:
        return False
    merged_polygons = vector_features.merge_touched_polygons(polygons, adjacent_matrix)
    print('finish merging touched polygons, get %d ones' % (len(merged_polygons)))

    # save
    wkt = map_projection.get_raster_or_vector_srs_info_wkt(in_shp)
    merged_pd = pd.DataFrame({'Polygon': merged_polygons})
    merged_shp = io_function.get_name_by_adding_tail(in_shp, 'merged')
    vector_gpd.save_polygons_to_files(merged_pd, 'Polygon', wkt, merged_shp)

def merge_based_gdal_rasterize(in_shp):

    # to raster
    # gdal_rasterize -b 1 -b 2 -b 3 -burn 255 -burn 0 -burn 0 -l mask mask.shp work.tif
    layer_name = os.path.splitext(os.path.basename(in_shp))[0]
    out_tif = layer_name + '.tif'
    command_str = 'gdal_rasterize -burn 255 -l ' + layer_name \
                    + ' -tr 30 30 ' + in_shp + ' ' + out_tif
    print(command_str)
    os.system(command_str)

    # set 0 as nodata
    command_str = ' gdal_edit.py -a_nodata 0 ' + out_tif
    os.system(command_str)

    # polygons
    out_shp = os.path.basename(io_function.get_name_by_adding_tail(in_shp,'merged_2'))
    command_string = 'gdal_polygonize.py -8 %s -b 1 -f "ESRI Shapefile" %s' % (out_tif, out_shp)
    print(command_string)
    os.system(command_str)


def main():
    # result of exp2
    # dir = os.path.expanduser('~/Data/Arctic/alaska/autoMapping/alaskaNS_yolov4_2/result_backup/alaska_north_slope_hillshade_2010to2017_alaskaNS_yolov4_2_exp2_1')
    # in_shp = os.path.join(dir,'alaska_north_slope_hillshade_2010to2017_alaskaNS_yolov4_2_exp2_post_1.shp')

    # merge_based_on_adjacent_matrix(in_shp)

    # result of exp1
    dir = os.path.expanduser('~/Data/Arctic/alaska/autoMapping/alaskaNS_yolov4_1/result_backup/alaska_north_slope_hillshade_2010to2017_alaskaNS_yolov4_1_exp1_1')
    in_shp = os.path.join(dir,'alaska_north_slope_hillshade_2010to2017_alaskaNS_yolov4_1_exp1_post_1.shp')


    merge_based_gdal_rasterize(in_shp)

if __name__ == '__main__':
    main()
    pass




