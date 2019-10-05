#!/usr/bin/env python
# Filename: download_planet_img 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 05 October, 2019
"""

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
from shapely.geometry import shape

from datetime import datetime
import json
import time


from planet import api
from planet.api import filters
# ClientV1 provides basic low-level access to Planet’s API. Only one ClientV1 should be in existence for an application.
client = None # api.ClientV1(api_key="abcdef0123456789")  #

asset_types=['analytic_sr','analytic_xml','udm']  # surface reflectance, metadata, mask file

downloaed_scene_geometry = []       # the geometry (extent) of downloaded images
manually_excluded_scenes = []       # manually excluded item id

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

def get_a_filter_cli_api(polygon_json,start_date, end_date, could_cover_thr):
    '''
    create a filter based on a geometry, date range, cloud cover
    :param polygon_json: a polygon in json format
    :param start_date: start date
    :param end_date:  end date
    :param could_cover_thr: images with cloud cover less than this value
    :return:  a combined filter (and filter)
    '''

    # gt: Greater Than
    # gte: Greater Than or Equal To
    # lt: Less Than
    # lte: Less Than or Equal To

    geo_filter = filters.geom_filter(polygon_json)
    date_filter = filters.date_range('acquired', gte=start_date, lte = end_date)
    cloud_filter = filters.range_filter('cloud_cover', lte=could_cover_thr)

    combined_filters = filters.and_filter(geo_filter, date_filter, cloud_filter)

    return combined_filters


def get_items_count(combined_filter, item_types):
    '''
    based on the filter, and item types, the count of item
    :param combined_filter: filter
    :param item_types: item types
    :return: the count of items
    '''

    req = filters.build_search_request(combined_filter, item_types, interval="year") #year  or day
    stats = client.stats(req).get()
    # p(stats)
    total_count = 0
    for bucket in stats['buckets']:
        total_count += bucket['count']
    return total_count


def activate_and_download_asset(item,asset_key,save_dir):
    '''
    active a asset of a item and download it
    :param item: the item
    :param asset_key: the name of the asset
    :param save_dir: save dir
    :return: True if successful, Flase otherwise
    '''

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
    body.await()

    return True

def read_down_load_geometry(folder):
    '''
    read geojson files in a folder. geojson file stores the geometry of a file, and save to global varialbes
    :param folder: the save folder
    :return:
    '''
    global  downloaed_scene_geometry
    json_list = io_function.get_file_list_by_ext('.geojson',folder, bsub_folder=False)
    for json_file in json_list:

        # ignore the scenes in the excluded list
        item_id = os.path.splitext(os.path.basename(json_file))[0]
        if item_id in manually_excluded_scenes:
            continue

        with open(json_file) as json_file:
            data = json.load(json_file)
            # p(data) # test
            downloaed_scene_geometry.append(data)

def read_excluded_scenes(folder):
    '''
    manually excluded some scenes with small portion of cloud cover,
    because some of the scenes have cloud cover, but not shown in the metedata
    :param folder:
    :return:
    '''
    txt_path = os.path.join(folder,'manually_excluded_scenes.txt')
    global manually_excluded_scenes
    if os.path.isfile(txt_path):
        with open(txt_path,'r') as f_obj:
            lines = f_obj.readlines()
            for line in lines:
                if '#' in line or len(line) < 6:
                    continue
                manually_excluded_scenes.append(line.strip())
    else:
        basic.outputlogMessage('Warning, %s file not exist'%txt_path)


def check_geom_polygon_overlap(boundary_list, polygon):
    '''
    check if a polygon is covered by any polygons in a geom_list
    :param boundary_list:  a list containing polygon
    :param polygon: a polygon
    :return: True if the polygon was cover a polyon by other, False otherwise
    '''

    # convert from json format to shapely
    polygon_shapely = shape(polygon)

    # using shapely to check the overlay
    for geom in boundary_list:
        geom_shapely = shape(geom)
        if geom_shapely.contains(polygon_shapely):
            return True

    return False

def download_planet_images(polygons_json, start_date, end_date, could_cover_thr, item_types, save_folder):
    '''
    download images from for all polygons, to save quota, each polygon only downlaod one image
    :param polygons_json: a list of polygons in json format
    :param start_date:
    :param end_date:
    :param could_cover_thr:
    :param save_folder:
    :return: True if successful, false otherwise
    '''

    for idx, geom in enumerate(polygons_json):

        # for test
        if idx > 20: break

        ####################################
        #check if any image already cover this polygon, if yes, skip downloading
        if check_geom_polygon_overlap(downloaed_scene_geometry, geom) is True:
            basic.outputlogMessage('%dth polygon already in the extent of download images, skip it')
            continue


        # search and donwload using Planet Client API
        combined_filter = get_a_filter_cli_api(geom, start_date, end_date, could_cover_thr)

        # get the count number
        item_count = get_items_count(combined_filter, item_types)
        basic.outputlogMessage('The total count number is %d' % item_count)

        req = filters.build_search_request(combined_filter, item_types)
        p(req)
        res = client.quick_search(req)
        if res.response.status_code == 200:
            all_items = []
            for item in res.items_iter(item_count):
                # print(item['id'], item['properties']['item_type'])
                all_items.append(item)

            # sort the item based on cloud cover
            all_items.sort(key=lambda x: float(x['properties']['cloud_cover']))
            # [print(item['id'],item['properties']['cloud_cover']) for item in all_items]

            # active and download them, only download the SR product
            download_item = all_items[0]
            for item in all_items:
                print(item['id'])
                if item['id'] not in manually_excluded_scenes:
                    download_item = item
                    break
                # assets = client.get_assets(item).get()
                # for asset in sorted(assets.keys()):
                #     print(asset)

            # I want to download SR, level 3B, product

            download_item_id = download_item['id']
            # p(item['geometry'])
            save_dir = os.path.join(save_folder, download_item_id)
            os.system('mkdir -p ' + save_dir)
            assets = client.get_assets(download_item).get()
            for asset in sorted(assets.keys()):
                if asset not in asset_types:
                    continue

                # activate and download
                activate_and_download_asset(download_item, asset, save_dir)

            # save the geometry of this item to disk
            with open(os.path.join(save_folder,download_item_id+'.geojson'), 'w') as outfile:
                json.dump(download_item['geometry'], outfile,indent=2)
                # update the geometry of already downloaded geometry
                downloaed_scene_geometry.append(download_item['geometry'])

        else:
            print('code {}, text, {}'.format(res.response.status_code, res.response.text))

    return True

def main(options, args):

    polygons_shp = args[0]
    save_folder = args[1]  # folder for saving downloaded images

    # check training polygons
    assert io_function.is_file_exist(polygons_shp)
    os.system('mkdir -p ' + save_folder)

    item_types = options.item_types.split(',') # ["PSScene4Band"]  # , # PSScene4Band , PSOrthoTile

    start_date = datetime.strptime(options.start_date, '%Y-%m-%d') #datetime(year=2018, month=5, day=20)
    end_date = datetime.strptime(options.end_date, '%Y-%m-%d')  #end_date
    could_cover_thr = options.cloud_cover           # 0.3

    planet_account = options.planet_account

    # set Planet API key
    get_and_set_Planet_key(planet_account)

    # read polygons
    polygons_json = read_polygons_json(polygons_shp)

    read_excluded_scenes(save_folder)   # read the excluded_scenes before read download images

    #read geometry of images already in "save_folder"
    read_down_load_geometry(save_folder)


    # download images
    download_planet_images(polygons_json, start_date, end_date, could_cover_thr, item_types, save_folder)



    test = 1



    pass

if __name__ == "__main__":

    usage = "usage: %prog [options] polygon_shp save_dir"
    parser = OptionParser(usage=usage, version="1.0 2019-10-01")
    parser.description = 'Introduction: search and download Planet images '
    parser.add_option("-s", "--start_date",default='2018-04-30',
                      action="store", dest="start_date",
                      help="start date for inquiry, with format year-month-day, e.g., 2018-05-23")
    parser.add_option("-e", "--end_date",default='2018-06-30',
                      action="store", dest="end_date",
                      help="the end date for inquiry, with format year-month-day, e.g., 2018-05-23")
    parser.add_option("-c", "--cloud_cover",
                      action="store", dest="cloud_cover", type=float,
                      help="the could cover threshold, only accept images with cloud cover less than the threshold")
    parser.add_option("-i", "--item_types",
                      action="store", dest="item_types",default='PSScene4Band',
                      help="the item types, e.g., PSScene4Band,PSOrthoTile")
    parser.add_option("-a", "--planet_account",
                      action="store", dest="planet_account",default='huanglingcao@link.cuhk.edu.hk',
                      help="planet email account, e.g., huanglingcao@link.cuhk.edu.hk")



    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)
