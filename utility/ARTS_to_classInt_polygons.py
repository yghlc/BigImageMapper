#!/usr/bin/env python
# Filename: ARTS_to_classInt_polygons.py 
"""
introduction: convert the data from of ARTS (https://github.com/whrc/ARTS) to
shapefile only containing class_int

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 20 February, 2025
"""

import os,sys
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)

import pandas as pd

import datasets.vector_gpd as vector_gpd
import basic_src.map_projection as map_projection
import basic_src.io_function as io_function


def ARTS_to_classInt_polygons(input,output,buff_radius=500):
    # read the existing data, convert to points
    geometries, train_class = vector_gpd.read_polygons_attributes_list(input, 'TrainClass',
                                                                       b_fix_invalid_polygon=False)
    print(f'read {len(geometries)} features')
    train_class_int = [1 if tclass == 'Positive' else 0 for tclass in train_class ]
    geom_centrioid_list = [vector_gpd.get_polygon_centroid(item) for item in geometries]

    geom_circle_list = [item.buffer(buff_radius) for item in geom_centrioid_list]


    id_list = [idx+1 for idx in range(len(train_class_int)) ]

    save_polyons_attributes = {'Polygons':geom_circle_list, 'id': id_list, 'class_int':train_class_int}
    wkt_string = map_projection.get_raster_or_vector_srs_info_proj4(input)
    polygon_df = pd.DataFrame(save_polyons_attributes)
    vector_gpd.save_polygons_to_files(polygon_df, 'Polygons', wkt_string, output, format='GPKG')


def main(options, args):
    input = args[0]
    output = options.save_path
    if output is None:
        output = io_function.get_name_by_adding_tail(input,'ClassInt')
    ARTS_to_classInt_polygons(input,output)
    pass


if __name__ == '__main__':
    usage = "usage: %prog [options] random1.shp "
    parser = OptionParser(usage=usage, version="1.0 2025-02-20")
    parser.description = 'Introduction: convert dataformat from ARTS to simpley polygons'

    parser.add_option("-s", "--save_path",
                      action="store", dest="save_path",
                      help="the file path for saving the results")


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)
