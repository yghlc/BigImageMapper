#!/usr/bin/env python
# Filename: polygons2geojson.py 
"""
introduction: convert polygons in shapefiles to many geojson (each for one polygon)

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 03 May, 2022
"""
import sys,os
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)

import basic_src.io_function as io_function
import basic_src.basic as basic
import basic_src.map_projection as map_projection
from datasets import raster_io
from datasets import vector_gpd

import geopandas as gpd
import pandas as pd

import rasterio
import json


def save_one_polygon_2geojson(poly, id, prj,save_folder):
    save_name = 'id_%d.geojson'%id
    save_path = os.path.join(save_folder,save_name)
    if os.path.isfile(save_path):
        basic.outputlogMessage('warning, %s exists, will be overwritten'%save_path)

    poly_info = {'Polygon':[poly]}
    save_pd = pd.DataFrame(poly_info)
    vector_gpd.save_polygons_to_files(save_pd, 'Polygon', prj, save_path,format='GeoJSON')

def polygons2geojson(input_shp,save_folder):
    '''
    convert polygons in shapefiles to many geojson (each for one polygon)
    :param input_shp:
    :param save_folder:
    :return:
    '''
    io_function.is_file_exist(input_shp)
    if os.path.isdir(save_folder) is False:
        io_function.mkdir(save_folder)

    polygons, ids = vector_gpd.read_polygons_attributes_list(input_shp,'id')
    prj_info = map_projection.get_raster_or_vector_srs_info_epsg(input_shp) # geojson need EPSG, such as "EPSG:3413"
    # print(prj_info)
    for poly, id in zip(polygons, ids):
        save_one_polygon_2geojson(poly, id,prj_info,save_folder)

def main(options, args):
    polygons2geojson(args[0],args[1])


def test_polygons2geojson():
    shp = os.path.expanduser('~/Data/Arctic/alaska/autoMapping/alaskaNS_yolov4_5/result_backup/'
                             'alaskaNotNS_hillshadeHWline_2008to2017_alaskaNS_yolov4_5_exp5_1/alaskaNotNS_hillshadeHWline_2008to2017_alaskaNS_yolov4_5_exp5_post_1_select_manuSelect.shp')

    polygons2geojson(shp, 'geojsons')


if __name__ == '__main__':
    usage = "usage: %prog [options] polygons_shp save_folder"
    parser = OptionParser(usage=usage, version="1.0 2021-4-15")
    parser.description = 'Introduction: convert polygons in shapefiles to many geojson (each for one polygon).'


    # parser.add_option("-p", "--para_file",
    #                   action="store", dest="para_file",
    #                   help="the parameters file")

    # test_polygons2geojson()

    (options, args) = parser.parse_args()
    # print(options.no_label_image)
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)
