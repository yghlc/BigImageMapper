#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Introduction: 
for some mapping or detection results, we may do not have ground truth to calculate the number of 
false negatives (missing detection),  To understand missing detection, we can check existing polygons 
and see how many of them were missed in the detection or mapping results. 
This script is used to check missing detection results.

Author: Huang Lingcao
Email: huanglingcao@gmail.com
Created: 2026-11-09
"""

import os, sys
from optparse import OptionParser

import pandas as pd

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
# import parameters
import basic_src.basic as basic
import basic_src.io_function as io_function
import basic_src.map_projection as map_projection
import datasets.vector_gpd as vector_gpd
import shapely
from shapely.strtree import STRtree


def check_missing_results(existing_data, mapping_results, buffer_size=500, b_existing_center=True, save_path=None, extent_shp=None):
    """
    check the missing detection results by comparing the existing data and mapping results
    :param existing_data: the existing data, a shapefile
    :param mapping_results: the mapping or detection results, a shapefile
    :param buffer_size: the buffer size for the existing data in meters
    :param b_existing_center: if set, will get the use the gemetric center for buffering
    :param save_path: the file path for saving the results
    :param extent_shp: the extent shapefile to limit the analysis area
    :return:
    """

    # check if existing_data and mapping_results have the same mappping projections
    existing_prj = map_projection.get_raster_or_vector_srs_info_epsg(existing_data)
    mapping_prj = map_projection.get_raster_or_vector_srs_info_epsg(mapping_results)


    #TODO: this block need to improve, currently, it only read data from "ARTS_main_dataset_v.2.1.0.gpkg", 
    # and only read the geometric center then buffering. 
    # read the existing data, convert to points
    if os.path.basename(existing_data).startswith('ARTS'):
        print('Assuming the existing data is ARTS dataset, will read the geometric center of the polygons')
        geometries, train_class = vector_gpd.read_polygons_attributes_list(existing_data,'TrainClass',b_fix_invalid_polygon=False)
        print(f'read {len(geometries)} features')
        geometries_pos = [item for item, tclass in zip(geometries,train_class) if tclass=='Positive']
        print(f'read {len(geometries_pos)} positive features')
    else:
        geometries_pos = vector_gpd.read_polygons_gpd(existing_data, b_fix_invalid_polygon=False)
        print(f'read {len(geometries_pos)} features')

    # filter out the features outside the extent shapefile
    if extent_shp is not None:
        extent_prj = map_projection.get_raster_or_vector_srs_info_epsg(extent_shp)
        if extent_prj != existing_prj:
            raise ValueError(f'the map projection is different between {existing_data} and {extent_shp}')

        extent_geoms = vector_gpd.read_polygons_gpd(extent_shp, b_fix_invalid_polygon=False)
        if len(extent_geoms) < 1:
            raise ValueError(f'the extent shapefile {extent_shp} has no features')
        ext_tree = STRtree(extent_geoms)
        filtered_geometries_pos = []
        for idx, geom in enumerate(geometries_pos):
            in_extent_list =  ext_tree.query(geom)
            if len(in_extent_list) > 0:
                filtered_geometries_pos.append(geom)
        print(f'after filtering by extent, {len(filtered_geometries_pos)} features left')
        geometries_pos = filtered_geometries_pos

    # print(geometries_pos[0])
    geom_pos_centrioid_list = [ vector_gpd.get_polygon_centroid(item) for item in geometries_pos]
    print(f'read {len(geom_pos_centrioid_list)} point features')
    geom_pos_circle_list = [item.buffer(buffer_size) for item in geom_pos_centrioid_list]


    if existing_prj != mapping_prj:
        # raise ValueError(f'the map projection is different between {existing_data} and {mapping_results}')
        basic.outputlogMessage(f'Warning, repoject the shapefile {os.path.basename(mapping_results)} to {existing_prj}')
        in_geometries = vector_gpd.read_shape_gpd_to_NewPrj(mapping_results,existing_prj)
    else:
        in_geometries = vector_gpd.read_polygons_gpd(mapping_results, b_fix_invalid_polygon=False)

    # build a tree
    tree = STRtree(in_geometries)

    # check each one in existing data, if it is in the mapping results
    b_overlap = []
    for idx, geom in enumerate(geom_pos_circle_list):
        inter_or_touch_list =  tree.query(geom)
        # print(inter_or_touch_list)
        if len(inter_or_touch_list) > 0:
            b_overlap.append(True)
        else:
            b_overlap.append(False)


    # output the results
    if save_path is None:
        existing_name = io_function.get_name_no_ext(existing_data)
        mapping_name = io_function.get_name_no_ext(mapping_results)
        save_path = existing_name + '_vs_' + mapping_name + '_missing.gpkg'

    # save b_overlap and geom_pos_circle_list to gpkg file, with the existing prj
    save_pd = pd.DataFrame({'Polygon': geom_pos_circle_list, 'overlap': b_overlap})
    vector_gpd.save_polygons_to_files(save_pd, 'Polygon', existing_prj, save_path,format='GPKG')

    # output brief information
    num_total = len(geom_pos_circle_list)
    num_overlap = sum(b_overlap)
    num_missing = num_total - num_overlap
    basic.outputlogMessage(f'the total number is {num_total}, '
                           f'missing: {num_missing}({num_missing/num_total*100:.2f}%), '
                           f'overlaping: {num_overlap}({num_overlap/num_total*100:.2f}%), saved to {save_path}')



    pass

def main(options, args):
    existing_data = args[0]
    mapping_results = args[1]
    buffer_size = options.buffer_size
    b_existing_center = options.b_existing_center
    save_path = options.save_path
    extent_shp = options.extent_shp

    check_missing_results(existing_data, mapping_results, buffer_size=buffer_size, b_existing_center=b_existing_center, save_path=save_path, extent_shp=extent_shp)




    pass


if __name__ == "__main__":

    usage = "usage: %prog [options] existing_data.shp mapping_results.shp "
    parser = OptionParser(usage=usage, version="1.0 2026-11-09")
    parser.description = 'Introduction: check the missing detection results by comparing the existing data and mapping results'

    parser.add_option("-s", "--save_path",
                      action="store", dest="save_path",
                      help="the file path for saving the results")

    parser.add_option("-b", "--buffer_size",
                      action="store", dest="buffer_size",
                      help="the buffer size for the existing data in meters, the default is 500", type=float, default=500)

    parser.add_option("-c", "--b_existing_center",
                      action="store_true", dest="b_existing_center",default=False,
                      help="if set, will get the use the gemetric center for buffering")

    parser.add_option("-e", "--extent_shp",
                      action="store", dest="extent_shp",
                      help="the extent shapefile to limit the analysis area")


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)

