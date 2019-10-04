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
import time

import requests
from requests.auth import HTTPBasicAuth


from planet import api
from planet.api import filters
# ClientV1 provides basic low-level access to Planet’s API. Only one ClientV1 should be in existence for an application.
client = None # api.ClientV1(api_key="abcdef0123456789")  #

asset_types=['analytic_sr','analytic_xml','udm']  # surface reflectance, metadata, mask file

def p(data):
    print(json.dumps(data, indent=2))

def get_and_set_Planet_key(user_account):
    keyfile = HOME+'/.planetkey'
    with open(keyfile) as f_obj:
        lines = f_obj.readlines()
        for line in lines:
            if user_account in line:
                key_str = line.split(':')[1]
                key_str = key_str.strip()       # remove '\n'
                os.environ["PL_API_KEY"] = key_str
                # set Planet API client
                global client
                client = api.ClientV1(api_key = key_str)

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

def get_a_filter_cli_api(polygon_json,start_date, end_date, could_cover_thr):

    geo_filter = filters.geom_filter(polygon_json)
    date_filter = filters.date_range('acquired', gte=start_date, lte = end_date)
    cloud_filter = filters.range_filter('cloud_cover', lte=could_cover_thr)

    combined_filters = filters.and_filter(geo_filter, date_filter, cloud_filter)

    return combined_filters

def get_items_count(combined_filter, item_types):

    req = filters.build_search_request(combined_filter, item_types, interval="year") #year  or day
    stats = client.stats(req).get()
    # p(stats)
    total_count = 0
    for bucket in stats['buckets']:
        total_count += bucket['count']
    return total_count


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

def active_and_downlaod_asset(item,asset_key,save_dir):

    assets = client.get_assets(item).get()

    asset = assets.get(asset_key)

    # activate
    activation = client.activate(asset)

    print(activation.response.status_code)

    asset_activated = False

    while asset_activated == False:
        # Get asset and its activation status
        assets = client.get_assets(item).get()  # need to used client to get the status
        asset = assets.get(asset_key)
        asset_status = asset["status"]

        # If asset is already active, we are done
        if asset_status == 'active':
            asset_activated = True
            print("Asset is active and ready to download")

        # Still activating. Wait and check again.
        else:
            print("...Still waiting for asset activation...")
            time.sleep(3)

    output_stream = sys.stdout
    def download_progress(start=None,wrote=None,total=None, finish=None): #result,skip=None
        # print(start,wrote,total,finish)
        # if total:
        #     # print('received: %.2f K'%(float(total)/1024.0))
        #     output_stream.write('received: %.2f K'%(float(total)/1024.0))
        #     output_stream.flush()
        # if total:
        #     if finish is None:
        #         print('received: %.2f K'%(float(total)/1024.0), end='\r')
        #     else:
        #         print('received: %.2f K' % (float(total) / 1024.0))
        pass
    callback = api.write_to_file(directory=save_dir + '/', callback=download_progress) # save_dir + '/'  #
    body = client.download(assets[asset_key], callback=callback)
    # if body._body.name == '':
    #     basic.outputlogMessage('Warning, the body name is missed, set as the asset key and id: item id: %s, asset %s'%(item['id'],asset_key))
    #     body._body.name = item['id']+'_'+asset_key  # AttributeError: can't set attribute
    body.await()

    return True


def main(options, args):

    # need to set the key first, and start API client
    get_and_set_Planet_key('huanglingcao@link.cuhk.edu.hk')
    # get_and_set_Planet_key('liulin@cuhk.edu.hk')
    # print(os.environ['PL_API_KEY'])

    # list_ItemTypes()

    polygons_shp = args[0]
    save_folder = args[1]  # folder for saving downloaded images

    # check training polygons
    assert io_function.is_file_exist(polygons_shp)
    os.system('mkdir -p ' + save_folder)

    polygons_json = read_polygons_json(polygons_shp)

    item_type = 'PSScene4Band'  # PSScene4Band , PSOrthoTile
    start_date = datetime.datetime(year=2018, month=5, day=20)
    end_date = datetime.datetime(year=2018, month=6, day=1)
    could_cover_thr = 0.3

    # ############################################################################################################
    # # search and download images using http request
    # # search_image_stats_for_a_polygon(polygons_json[0], item_type, start_date, end_date, could_cover_thr)
    #
    # images_ids = search_image_metadata_for_a_polygon(polygons_json[0], item_type, start_date, end_date, could_cover_thr)
    # [print(item) for item in images_ids ]
    #
    # asset_type_list = get_asset_type(item_type, images_ids[1])
    # [print(item) for item in asset_type_list]
    #
    # for asset_type in asset_type_list:
    #     download_url = activation_a_item(images_ids[0], item_type, asset_type)
    #
    #     if download_url is None:
    #         basic.outputlogMessage('failed to get the location of %s'%asset_type )
    #         continue
    #
    #     # download ehte activated item
    #     if 'xml' == asset_type.split('_')[-1]:
    #         output = images_ids[0] + '_' + asset_type  + '.xml'
    #     elif 'rpc' == asset_type.split('_')[-1]:
    #         output = images_ids[0] + '_' + asset_type + '.txt'
    #     else:
    #         output = images_ids[0] + '_' + asset_type + '.tif'
    #     # images_ids[0]+'.tif'
    #     download_one_item(download_url,os.path.join(save_folder,output))
    ############################################################################################################
    # search and donwload using Planet Client API
    # p(polygons_json[0]) # print a polygon in JSON format
    geom = polygons_json[0]
    combined_filter = get_a_filter_cli_api(geom, start_date, end_date, could_cover_thr)
    # p(combined_filter)

    item_types = ["PSScene4Band"] #, "PSOrthoTile"

    # get the count number
    item_count = get_items_count(combined_filter, item_types)
    basic.outputlogMessage('The total count number is %d'%item_count)

    req = filters.build_search_request(combined_filter, item_types)
    p(req)
    res = client.quick_search(req)
    if res.response.status_code == 200:
        # print('good')
        all_items = []
        for item in res.items_iter(item_count):
            # print(item['id'], item['properties']['item_type'])
            all_items.append(item)

        # sort the item based on cloud cover
        all_items.sort(key=lambda x: float(x['properties']['cloud_cover']))
        # [print(item['id'],item['properties']['cloud_cover']) for item in all_items]

        # active and download them, only download the SR product
        for item in all_items:
            print(item['id'])
            assets = client.get_assets(item).get()
            for asset in sorted(assets.keys()):
                print(asset)

        # I want to download SR, level 3B, product
        # test: download the first one, all
        # not all the item has "analytic_sr"
        item = all_items[0]
        assets = client.get_assets(item).get()
        for asset in sorted(assets.keys()):
            print(asset)
            if 'basic' in asset:
                print('skip %s'% asset)
                continue

            if asset not in asset_types:
                continue

            # active and download
            active_and_downlaod_asset(item, asset, save_folder)

        # active_and_downlaod_asset(item, 'analytic_sr', save_folder)
        # active_and_downlaod_asset(item, 'basic_analytic_dn_nitf', save_folder)
        # active_and_downlaod_asset(item, 'analytic_sr', save_folder)

    else:
        print('code {}, text, {}'.format(res.response.status_code, res.response.text))


    test = 1
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






