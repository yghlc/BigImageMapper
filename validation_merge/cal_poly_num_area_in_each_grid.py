#!/usr/bin/env python
# Filename: cal_poly_num_area_in_each_grid.py
"""
introduction: calculate the number and areas of different mapping results (polygon) within or intersect grids or cells


authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 02 September, 2025
"""

import os,sys
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)

import basic_src.io_function as io_function
import basic_src.basic as basic
import datasets.vector_gpd as vector_gpd

from datetime import datetime

import pandas as pd
import geopandas as gpd
from shapely.geometry import box

def calculate_poly_count_area_in_each_grid(grid_vector, in_poly_vector, save_path, column_pre_name='poly',
                                           b_poly_bounds = False):

    io_function.is_file_exist(grid_vector)
    io_function.is_file_exist(in_poly_vector)

    column_count = f"{column_pre_name}_C"
    column_area = f"{column_pre_name}_A"

    if vector_gpd.is_field_name_in_shp(save_path,column_count):
        print(f'Column {column_count} already in {save_path}, skip calculating')
        return

    grid_prj = vector_gpd.get_projection(grid_vector)
    in_poly_proj = vector_gpd.get_projection(in_poly_vector)
    if grid_prj != in_poly_proj:
        raise ValueError(f'The map projection between {grid_vector} and {in_poly_vector} is different')

    # print(grid_prj)

    if grid_prj == "EPSG:4326":
        raise ValueError('Not support lat/long projection')

    grid_gpd = gpd.read_file(grid_vector)
    poly_gpd = gpd.read_file(in_poly_vector)
    if b_poly_bounds:
        poly_gpd['geometry'] = poly_gpd.bounds.apply(lambda row: box(row.minx, row.miny, row.maxx, row.maxy), axis=1)

    print(poly_gpd)

    # Spatial join: for each grid, find intersecting polygons
    joined = gpd.sjoin(grid_gpd, poly_gpd, how="left", predicate="intersects")
    # print(joined)

    # For area calculation, get the intersection geometries
    joined["intersection"] = joined.apply(
        lambda row: row.geometry.intersection(poly_gpd.loc[row.index_right].geometry)
        if pd.notnull(row.index_right) else None, axis=1)

    # print(joined)


    joined["intersect_area"] = joined["intersection"].area
    # print(joined)

    # Group by grid cell index and aggregate
    grid_summary = (joined.groupby(joined.index).agg(poly_count=("index_right", lambda x: x.notnull().sum()),
             total_intersect_area=("intersect_area", "sum")))

    # Merge results back into grid
    grid_result = grid_gpd.copy()
    grid_result[column_count] = grid_summary["poly_count"]
    grid_result[column_area] = grid_summary["total_intersect_area"]

    # Fill NaNs with 0 (for grid cells with no intersections)
    grid_result = grid_result.fillna({column_count: 0, column_area: 0})

    # Save to desired path
    grid_result.to_file(save_path)
    print(datetime.now(),f'saved to {save_path}')


def test_calculate_poly_count_area_in_each_grid():

    # grid_vector=os.path.expanduser('~/Downloads/tmp/h3_cells_for_merging/h3_cells_res8_panArctic_s2_rgb_2024_object_detection_s2_exp5_post_1.shp')
    grid_vector=os.path.expanduser('~/Downloads/tmp/h3_cells_panArctic_rts.gpkg')
    in_poly_vector = os.path.expanduser('~/Data/published_data/Dai_etal_2025_largeRTS_ArcticDEM/ArcticRTS_epsg3413_ClassInt.shp')

    save_path = 'polygon_count_area_per_grid.gpkg'
    calculate_poly_count_area_in_each_grid(grid_vector, in_poly_vector, save_path,b_poly_bounds=True)



def main(options, args):

    grid_vector = args[0]
    # in_vectors = [item for item in args[1:]]
    # save_path = options.save_path
    save_path = grid_vector   # save the result into the orignal grid
    b_using_bounding_box = options.using_bounding_box

    input_txt = options.input_txt
    in_vectors_colum_dict = {}
    if input_txt is not None:
        tmp_list = io_function.read_list_from_txt(input_txt)
        for tmp in tmp_list:
            col_and_file = [item.strip() for item in tmp.split(',')]
            # print(col_and_file)
            in_vectors_colum_dict[col_and_file[0]] = col_and_file[1]
    else:
        print('Please set "--input_txt"')
        return

    # if save_path != grid_vector:
    #     print('Please ')
    #     return


    # save for backup
    io_function.save_dict_to_txt_json(input_txt+'.json',in_vectors_colum_dict)

    for idx, col_name in enumerate(in_vectors_colum_dict.keys()):
        in_poly_vector = in_vectors_colum_dict[col_name]
        basic.outputlogMessage(f'({idx+1}/{len(in_vectors_colum_dict)}) Working on {in_vectors_colum_dict[col_name]}')

        # print(grid_vector, in_poly_vector, save_path, col_name,b_using_bounding_box)

        calculate_poly_count_area_in_each_grid(grid_vector, in_poly_vector, save_path, column_pre_name=col_name,
                                               b_poly_bounds=b_using_bounding_box)



if __name__ == '__main__':

    # test_calculate_poly_count_area_in_each_grid()
    # sys.exit(0)

    usage = "usage: %prog [options] grid_vector "
    parser = OptionParser(usage=usage, version="1.0 2021-4-15")
    parser.description = 'Introduction: convert polygons in shapefiles to many geojson (each for one polygon).'

    # parser.add_option("-s", "--save_path",
    #                   action="store", dest="save_path",default="polygon_count_area_per_grid.gpkg",
    #                   help="the save path ")

    parser.add_option("-i", "--input_txt",
                      action="store", dest="input_txt",
                      help="the input txt contain column name and vector path (column_name, vector_path)")

    parser.add_option("-b", "--using_bounding_box",
                      action="store_true", dest="using_bounding_box",default=False,
                      help="whether use the boudning boxes of polygons, this can avoid some invalid"
                           " polygons and be consistent with YOLO output")


    (options, args) = parser.parse_args()
    # print(options.no_label_image)
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)
