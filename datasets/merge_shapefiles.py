#!/usr/bin/env python
# Filename: merge_shapefiles 
"""
introduction: Merge shape files (should have the same projection)
a similar version written in bash can be found in: ~/codes/PycharmProjects/Landuse_DL/sentinelScripts/merge_shapefiles.sh

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 08 June, 2020
"""


import os, sys

import pandas as pd
import geopandas as gpd
from optparse import OptionParser

import basic_src.basic as basic
from basic_src.map_projection import get_raster_or_vector_srs_info_proj4
import vector_gpd

def merge_shape_files(file_list, save_path):

    if os.path.isfile(save_path):
        print('%s already exists'%save_path)
        return True
    if len(file_list) < 1:
        raise IOError("no input shapefiles")

    ref_prj = get_raster_or_vector_srs_info_proj4(file_list[0])

    # read polygons as shapely objects
    attribute_names = None
    polygons_list = []
    polygon_attributes_list = []

    b_get_field_name = False

    for idx, shp_path in enumerate(file_list):

        # check projection
        prj = get_raster_or_vector_srs_info_proj4(file_list[idx])
        if prj != ref_prj:
            raise ValueError('Projection inconsistent: %s is different with the first one'%shp_path)

        shapefile = gpd.read_file(shp_path)
        if len(shapefile.geometry.values) < 1:
            basic.outputlogMessage('warning, %s is empty, skip'%shp_path)
            continue

        # go through each geometry
        for ri, row in shapefile.iterrows():
            # if idx == 0 and ri==0:
            if b_get_field_name is False:
                attribute_names = row.keys().to_list()
                attribute_names = attribute_names[:len(attribute_names) - 1]
                # basic.outputlogMessage("attribute names: "+ str(row.keys().to_list()))
                b_get_field_name = True

            polygons_list.append(row['geometry'])
            polygon_attributes = row[:len(row) - 1].to_list()
            if len(polygon_attributes) < len(attribute_names):
                polygon_attributes.extend([None]* (len(attribute_names) - len(polygon_attributes)))
            polygon_attributes_list.append(polygon_attributes)

    # save results
    save_polyons_attributes = {}
    for idx, attribute in enumerate(attribute_names):
        # print(idx, attribute)
        values = [item[idx] for item in polygon_attributes_list]
        save_polyons_attributes[attribute] = values

    save_polyons_attributes["Polygons"] = polygons_list
    polygon_df = pd.DataFrame(save_polyons_attributes)


    return vector_gpd.save_polygons_to_files(polygon_df, 'Polygons', ref_prj, save_path)


def main(options, args):

    folder = args[0]
    shp_list = io_function.get_file_list_by_ext('.shp',folder,bsub_folder=False)

    # remove I*
    out_name_list = [ '_'.join(os.path.basename(shp).split('_')[1:]) for shp in shp_list ]
    # remove duplicated ones
    out_name_list = [ item for item in set(out_name_list) ]

    for out_name in out_name_list:
        file_list = [ item for item in shp_list if out_name in item ]
        # no need to remove "out_name" if it exist, it will be overwrite
        merge_shape_files(file_list, out_name)

    pass

if __name__ == "__main__":

    sys.path.insert(0, os.path.expanduser('~/codes/PycharmProjects/DeeplabforRS'))

    import basic_src.io_function as io_function
    import basic_src.basic as basic

    from basic_src.map_projection import get_raster_or_vector_srs_info_proj4
    import vector_gpd

    usage = "usage: %prog [options] folder "
    parser = OptionParser(usage=usage, version="1.0 2020-07-27")
    parser.description = 'Introduction: merge shape files '


    # parser.add_option("-p", "--para",
    #                   action="store", dest="para_file",
    #                   help="the parameters file")

    # parser.add_option('-o', '--output',
    #                   action="store", dest = 'output',
    #                   help='the path to save the change detection results')

    (options, args) = parser.parse_args()
    # if len(sys.argv) < 2:
    #     parser.print_help()
    #     sys.exit(2)



    main(options, args)


    pass