#!/usr/bin/env python
# Filename: clear_polygons.py 
"""
introduction: filter the polygons after crowdsourcing validation

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 26 September, 2022
"""

import os,sys
from optparse import OptionParser

import pandas as pd
import re
import json

deeplabforRS = os.path.expanduser('~/codes/PycharmProjects/DeeplabforRS')
sys.path.insert(0, deeplabforRS)
import vector_gpd
import basic_src.io_function as io_function
import basic_src.basic as basic


# a dict of user name and ids
# user_names = {}     # user_id <-> user_name

# def read_user_names(user_json_file):
#     # read json files, to many dict
#     with open(user_json_file) as f_obj:
#         data_list = json.load(f_obj)
#
#     for rec in data_list:
#         user_names[rec['pk']] = rec['fields']['username']       # id : username


def read_a_geojson_latlon(file_path):
    polygons = vector_gpd.read_shape_gpd_to_NewPrj(file_path,'EPSG:4326')
    # print(polygons)
    # only keep the one with Polygon type
    check_polygons = []
    for poly in polygons:
        # print(poly.type,type(poly.type))
        if poly.type == 'Polygon':
            check_polygons.append(poly)
        else:
            basic.outputlogMessage('Warning, ignore one geometry which is not Polygon but %s in %s '%(poly.type, file_path))

    return check_polygons


def test_read_a_geojson_latlon():
    # geojson = '/Users/huanglingcao/Data/labelearth.colorado.edu/data/data/objectPolygons/img000001_panArctic_time0_poly_3028.geojson'
    # geojson = '/Users/huanglingcao/Data/labelearth.colorado.edu/data/data/objectPolygons/img000001_panArctic_time0_poly_3028_by_CarolXU.geojson'
    geojson = '/Users/huanglingcao/Data/labelearth.colorado.edu/data/data/objectPolygons/img000010_panArctic_time0_poly_432_by_liulin@cuhk.edu.hk_000.geojson'
    read_a_geojson_latlon(geojson)

def merge_inputs_from_users(userinput_json_file,dir_geojson,user_json_file,image_json,save_path):
    with open(userinput_json_file) as f_obj:
        userinput_list = json.load(f_obj)

    with open(user_json_file) as f_obj:
        user_list = json.load(f_obj)

    with open(image_json) as f_obj:
        image_json = json.load(f_obj)

    user_names = {}
    for rec in user_list:
        user_names[rec['pk']] = rec['fields']['username']       # id : username

    image_polygon_file = {}
    for rec in image_json:
        image_polygon_file[rec['pk']] = rec['fields']['image_object_path']       # image id : path to bounding boxes

    image_polygons_valid_res = {}
    for rec in userinput_list:
        image_polygons_valid_res.setdefault('input_pk', []).append(rec['pk'])
        image_polygons_valid_res.setdefault('image_pk', []).append(rec['fields']['image_name'])
        image_polygons_valid_res.setdefault('user_pk', []).append(rec['fields']['user_name'])
        org_json = os.path.basename(image_polygon_file[rec['fields']['image_name']])
        image_polygons_valid_res.setdefault('org_polygon', []).append(org_json)    # original polygons
        image_polygons_valid_res.setdefault('user_name', []).append(user_names[ rec['fields']['user_name']])
        image_polygons_valid_res.setdefault('possibility', []).append(rec['fields']['possibility'])
        image_polygons_valid_res.setdefault('user_note', []).append(rec['fields']['user_note'])
        image_polygons_valid_res.setdefault('user_polygon', []).append(rec['fields']['user_image_output'])          # modified or added polygons


     # save to excel file
    # save_xlsx = io_function.get_name_no_ext(save_path) + '_all_records.xlsx'
    # table_pd = pd.DataFrame(image_polygons_valid_res)
    # with pd.ExcelWriter(save_xlsx) as writer:
    #     table_pd.to_excel(writer)


    # convert and save to shapefile
    user_name_list = []
    possibility_list = []
    org_polygon_name_list = []
    user_note_list = []
    polygon_list = []

    for user_name, possibility,user_note, org_json,user_input_json in zip(image_polygons_valid_res['user_name'],
                                                                          image_polygons_valid_res['possibility'],
                                                                          image_polygons_valid_res['user_note'],
                                                                          image_polygons_valid_res['org_polygon'],
                                                                          image_polygons_valid_res['user_polygon']):
        # add original polygons
        # print(org_json)
        org_json_path = os.path.join(dir_geojson, org_json)
        polygon_list.append(read_a_geojson_latlon(org_json_path)[0])
        user_name_list.append(user_name)
        org_polygon_name_list.append(org_json)
        possibility_list.append(possibility)
        user_note_list.append(user_note)

        if user_input_json is not None:
            # print(user_input_json)
            user_input_json_path = os.path.join(dir_geojson, os.path.basename(user_input_json) )
            new_polys = read_a_geojson_latlon(user_input_json_path)
            for n_poly in new_polys:
                polygon_list.append(n_poly)
                user_name_list.append(user_name)
                org_polygon_name_list.append(org_json)
                possibility_list.append('useradd')      # these are addded by users
                user_note_list.append(user_note)    # could be duplicated, but ok

    # save to file
    polygon_pd = pd.DataFrame({'user':user_name_list,'o_geojson':org_polygon_name_list, 'possibilit':possibility_list,
                               'note':user_note_list,'polygons':polygon_list})
    vector_gpd.save_polygons_to_files(polygon_pd, 'polygons', 'EPSG:4326', save_path,format='ESRI Shapefile')



def main(options, args):
    userinput_json = args[0]
    dir_geojson = args[1]
    save_path = options.save_path
    user_json = options.user_json
    image_json = options.image_json

    io_function.is_file_exist(userinput_json)
    io_function.is_folder_exist(dir_geojson)
    if save_path is None:
        save_path = 'polygons_after_webValidation.shp'

    # merge polygons
    before_filter_save = io_function.get_name_by_adding_tail(save_path,'NoFilter')
    merge_inputs_from_users(userinput_json,dir_geojson, user_json,image_json,before_filter_save)

    # filter polygons





if __name__ == '__main__':
    usage = "usage: %prog [options] userinput.json dir_objectPolygons"
    parser = OptionParser(usage=usage, version="1.0 2022-09-26")
    parser.description = 'Introduction: clear polygons after web-based validation '

    parser.add_option("-s", "--save_path",
                      action="store", dest="save_path",
                      help="the path for saving file")

    parser.add_option("-u", "--user_json",
                      action="store", dest="user_json",
                      help="the json file containing user information, exported by Django")

    parser.add_option("-i", "--image_json",
                      action="store", dest="image_json",
                      help="the json file containing image information, exported by Django")


    (options, args) = parser.parse_args()
    # print(options.create_mosaic)

    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)
