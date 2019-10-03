#!/usr/bin/env python
# Filename: autoDownload_PlanetImg 
"""
introduction: download Planet image via Planet API downloader

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 30 September, 2019
"""

### copied from thesis: "Detection and analysis of thermokarst related landscape processes using temporally and spatially high-resolution Planet Cube Sat data"
# Data acquisition:
# The data acquisition using the Planet API downloader was simple and effective.
# The required data could be downloaded after adapting the tutorial for the personal needs.
# It is necessary to be careful which type of data you acquire, as there are different possibilities
# which include the Basic scene, Ortho scenes as well as Ortho tile scene (for further information see 1.1.2).
# After the data type selection is done, you receive a download link which is valid for 5 minutes.
# By clicking on it, the download starts, and you can save the image to your device.


# pre-install library
# python 2.7+, better to use Python 3
# pip install requests
# pip install retrying
# pip install jq

import sys,os
from optparse import OptionParser

HOME = os.path.expanduser('~')

# path of DeeplabforRS
codes_dir2 = HOME + '/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)

import basic_src.io_function as io_function
import basic_src.basic as basic

# import thest two to make sure load GEOS dll before using shapely
import shapely
from shapely.geometry import mapping # transform to GeJSON format
import geopandas as gpd

import datetime
import json

import requests
from requests.auth import HTTPBasicAuth

def get_and_set_Planet_key(user_account):
    keyfile = HOME+'/.planetkey'
    with open(keyfile) as f_obj:
        lines = f_obj.readlines()
        for line in lines:
            if user_account in line:
                key_str = line.split(':')[1]
                key_str = key_str.strip()       # remove '\n'
                os.environ["PL_API_KEY"] = key_str
                return True
        raise ValueError('account: %s cannot find in %s'%(user_account,keyfile))

def list_ItemTypes():

    # Each class of imagery is identified by its "ItemType".
    # e.g., "PSOrthoTile" - Images taken by PlanetScope satellites in the OrthoTile format.
    # e.g., "REOrthoTile" - Images taken by RapidEye satellites in the OrthoTile format.

    command_str = "curl -L -H \"Authorization: api-key $PL_API_KEY\" \'https://api.planet.com/data/v1/item-types\' | jq \'.item_types[].id\'"
    out_str = basic.exec_command_string_output_string(command_str)
    print(out_str)
    return out_str

def read_polygons_json(polygon_shp):
    '''
    read polyogns and convert to json format
    :param polygon_shp: polygon in projection of EPSG:4326
    :return:
    '''

    # check projection
    shp_args_list = ['gdalsrsinfo', '-o', 'EPSG', polygon_shp]
    epsg_str = basic.exec_command_args_list_one_string(shp_args_list)
    epsg_str = epsg_str.decode().strip()  # byte to str, remove '\n'
    if epsg_str != 'EPSG:4326':
        raise ValueError('Current support shape file in projection of EPSG:4326')

    shapefile = gpd.read_file(polygon_shp)
    polygons = shapefile.geometry.values

    # check invalidity of polygons
    invalid_polygon_idx = []
    for idx, geom in enumerate(polygons):
        if geom.is_valid is False:
            invalid_polygon_idx.append(idx + 1)
    if len(invalid_polygon_idx) > 0:
        raise ValueError('error, polygons %s (index start from 1) in %s are invalid, please fix them first '%(str(invalid_polygon_idx),polygon_shp))

    # convert to json format
    polygons_json = [ mapping(item) for item in polygons]

    return polygons_json

def get_a_filter(polygon_json,start_date, end_date, could_cover_thr):
    # filter for items the overlap with our chosen geometry
    geometry_filter = {
        "type": "GeometryFilter",
        "field_name": "geometry",
        "config": polygon_json
    }

    # filter images acquired in a certain date range
    date_range_filter = {
        "type": "DateRangeFilter",
        "field_name": "acquired",
        "config": {
            "gte": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),  # "2016-07-01T00:00:00.000Z"
            "lte": end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),  # "2016-08-01T00:00:00.000Z"
        }
    }

    # filter any images which are more than 50% clouds
    cloud_cover_filter = {
        "type": "RangeFilter",
        "field_name": "cloud_cover",
        "config": {
            "lte": could_cover_thr  # 0.5
        }
    }

    # create a filter that combines our geo and date filters
    # could also use an "OrFilter"
    combined_filters = {
        "type": "AndFilter",
        "config": [geometry_filter, date_range_filter, cloud_cover_filter]
    }

    return combined_filters

def POST_request(url, filter):
    print(os.environ['PL_API_KEY'])
    return requests.post(url,
            auth=HTTPBasicAuth(os.environ['PL_API_KEY'], ''),
            json=filter)

def search_image_stats_for_a_polygon(polygon_json, item_type, start_date, end_date, could_cover_thr):
    '''
    search images with a polyon as input
    :param polygon_json:
    :param item_type:
    :param period:
    :param could_cover:
    :return: buckets showing how many images available for each day
    '''

    combined_filters = get_a_filter(polygon_json, start_date, end_date, could_cover_thr)

    # Stats API request object
    stats_endpoint_request = {
        "interval": "day",
        "item_types": [item_type],
        "filter": combined_filters
    }

    # fire off the POST request
    result = POST_request('https://api.planet.com/data/v1/stats', stats_endpoint_request)

    print(result.text)

    # os.system('jq '+ result.text)
    result_dict = json.loads(result.text)
    buckets = result_dict['buckets']
    for bucket in buckets:
        print(bucket)

    return True

def search_image_metadata_for_a_polygon(polygon_json, item_type, start_date, end_date, could_cover_thr):
    '''
    search images with a polyon as input
    :param polygon_json:
    :param item_type:
    :param period:
    :param could_cover:
    :return: image metadata and the corresponding id
    '''

    combined_filters = get_a_filter(polygon_json, start_date, end_date, could_cover_thr)

    # Stats API request object
    stats_endpoint_request = {
        "item_types": [item_type],
        "filter": combined_filters
    }

    # fire off the POST request
    result = POST_request('https://api.planet.com/data/v1/quick-search', stats_endpoint_request)

    # print(result.text)

    result_dict = json.loads(result.text)
    # buckets = result_dict['buckets']
    # for bucket in buckets:
    #     print(bucket)
    print('********************************************************************')
    # print(result_dict['_links'])
    print('********************************************************************')
    ids = []
    for feature in result_dict['features']:
        ids.append(feature['id'])
        # print(feature['id'])
        # for key in feature.keys():
        #     print(key,':', feature[key])
        # break
        # print(feature['id'])
        # # return True

    return ids

def get_asset_type(item_type,item_id):

    url = 'https://api.planet.com/data/v1/item-types/%s/items/%s/assets'%(item_type,item_id)

    command_str = "curl -L -H \"Authorization: api-key $PL_API_KEY\" " + url
    out_str = basic.exec_command_string_output_string(command_str)
    asset_types = json.loads(out_str)
    # print(out_str)

    # test

    for key in asset_types.keys():
        # print(key)
        print(asset_types[key])

    return list(asset_types.keys())


def activation_a_item(item_id, item_type, asset_type):
    '''
    activate a item
    :param item_id:
    :param item_type:
    :param asset_type:
    :return:
    '''
    session = requests.Session()
    session.auth = (os.environ['PL_API_KEY'], '')

    # request an item
    item = session.get(
            ("https://api.planet.com/data/v1/item-types/" +
             "{}/items/{}/assets/").format(item_type, item_id))

    # extract the activation url from the item for the desired asset
    item_activation_url = item.json()[asset_type]["_links"]["activate"]

    # request activation
    response = session.post(item_activation_url)

    #A response of 202 means that the request has been accepted and the activation will begin shortly.
    # A 204 code indicates that the asset is already active and no further action is needed.
    #  A 401 code means the user does not have permissions to download this file.
    if response.status_code == 204:
        # success, return location
        url = 'https://api.planet.com/data/v1/item-types/%s/items/%s/assets'%(item_type,item_id)
        command_str = "curl -L -H \"Authorization: api-key $PL_API_KEY\" " + url
        out_str = basic.exec_command_string_output_string(command_str)
        tmp_dict = json.loads(out_str)
        print(tmp_dict[asset_type]['location'])

        return tmp_dict[asset_type]['location']



    print (response.status_code)

# def activation_a_item_url(url):
#     '''
#     activate a item by url
#     :param url:
#     :return:
#     '''
#     session = requests.Session()
#     session.auth = (os.environ['PL_API_KEY'], '')
#
#     # request an item
#     item = session.get(
#             ("https://api.planet.com/data/v1/item-types/" +
#              "{}/items/{}/assets/").format(item_type, item_id))
#
#     # extract the activation url from the item for the desired asset
#     item_activation_url = item.json()[asset_type]["_links"]["activate"]
#
#     # request activation
#     response = session.post(item_activation_url)
#
#     print (response.status_code)

def download_one_item(download_url, save_path):

    command_str = "curl -L " + download_url + ' > ' +save_path
    return basic.exec_command_string(command_str)


def main(options, args):

    # need to set the key first
    get_and_set_Planet_key('huanglingcao@link.cuhk.edu.hk')
    # print(os.environ['PL_API_KEY'])

    # list_ItemTypes()

    polygons_shp = args[0]
    save_folder = args[1]  # folder for saving downloaded images

    # check training polygons
    assert io_function.is_file_exist(polygons_shp)
    os.system('mkdir -p ' + save_folder)

    polygons_json = read_polygons_json(polygons_shp)

    item_type = 'PSOrthoTile'  # PSScene4Band , PSOrthoTile
    start_date = datetime.date(2018, 5, 20) # year, month, day
    end_date = datetime.date(2018, 6, 1)
    could_cover_thr = 0.3

    # search_image_stats_for_a_polygon(polygons_json[0], item_type, start_date, end_date, could_cover_thr)

    images_ids = search_image_metadata_for_a_polygon(polygons_json[0], item_type, start_date, end_date, could_cover_thr)
    [print(item) for item in images_ids ]

    asset_type_list = get_asset_type(item_type, images_ids[1])
    [print(item) for item in asset_type_list]

    for asset_type in asset_type_list:
        download_url = activation_a_item(images_ids[0], item_type, asset_type)

        if download_url is None:
            basic.outputlogMessage('failed to get the location of %s'%asset_type )
            continue

        # download ehte activated item
        if 'xml' == asset_type.split('_')[-1]:
            output = images_ids[0] + '_' + asset_type  + '.xml'
        elif 'rpc' == asset_type.split('_')[-1]:
            output = images_ids[0] + '_' + asset_type + '.txt'
        else:
            output = images_ids[0] + '_' + asset_type + '.tif'
        # images_ids[0]+'.tif'
        download_one_item(download_url,os.path.join(save_folder,output))

    pass

if __name__ == "__main__":

    usage = "usage: %prog [options] polygon_shp save_dir"
    parser = OptionParser(usage=usage, version="1.0 2019-10-01")
    parser.description = 'Introduction: search and download Planet images '
    parser.add_option("-f", "--all_training_polygons",
                      action="store", dest="all_training_polygons",
                      help="the full set of training polygons. If the one in the input argument "
                           "is a subset of training polygons, this one must be assigned")
    # parser.add_option("-b", "--bufferSize",
    #                   action="store", dest="bufferSize", type=float,
    #                   help="buffer size is in the projection, normally, it is based on meters")
    # parser.add_option("-e", "--image_ext",
    #                   action="store", dest="image_ext", default='.tif',
    #                   help="the extension of the image file")
    # parser.add_option("-o", "--out_dir",
    #                   action="store", dest="out_dir",
    #                   help="the folder path for saving output files")
    # parser.add_option("-n", "--dstnodata", type=int,
    #                   action="store", dest="dstnodata",
    #                   help="the nodata in output images")
    # parser.add_option("-r", "--rectangle",
    #                   action="store_true", dest="rectangle", default=False,
    #                   help="whether use the rectangular extent of the polygon")

    (options, args) = parser.parse_args()
    # if len(sys.argv) < 2 or len(args) < 1:
    #     parser.print_help()
    #     sys.exit(2)


    main(options, args)






