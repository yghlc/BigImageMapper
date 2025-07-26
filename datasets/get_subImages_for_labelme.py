#!/usr/bin/env python
# Filename: get_subImages_for_labelme.py
"""
introduction: get sub-images from images based on polygons (same to "get_subImages.py"),
            but convert the polygons in each sub-images to labelme Json format.

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 15 April, 2021
"""

import sys,os
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)

import parameters
import basic_src.io_function as io_function
import basic_src.basic as basic
from datasets import raster_io

import geopandas as gpd
import rasterio
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import get_subImages

from shapely.geometry import Polygon, MultiPolygon

# def get_sub_image(idx,selected_polygon, image_tile_list, image_tile_bounds, save_path, dstnodata, brectangle ):

def get_one_sub_image_json_file(idx, center_polygon, c_class_int,class_names, image_tile_list, image_tile_bounds, save_path, dstnodata, brectangle,
                                bufferSize, polygons_all, class_labels_all,out_format):
    '''

    :param idx:
    :param selected_polygon:
    :param c_class_int:
    :param image_tile_list:
    :param image_tile_bounds:
    :param save_path:
    :param dstnodata:
    :param brectangle:
    :param polygons_all:
    :param class_labels_all:
    :return:
    '''

    # get the sub images.
    expansion_polygon = center_polygon.buffer(bufferSize)
    get_subImages.get_sub_image(idx, expansion_polygon, image_tile_list,image_tile_bounds,save_path,dstnodata,
                                brectangle, False, out_format=out_format)

    #
    save_josn_path = os.path.splitext(save_path)[0] + '.json'

    # get adjacent polygon
    adj_polygons, adj_polygons_class = get_subImages.get_adjacent_polygons(center_polygon, polygons_all, class_labels_all, bufferSize,
                                                             brectangle)
    # add the center polygons to adj_polygons
    adj_polygons.extend([center_polygon])
    adj_polygons_class.extend([c_class_int])

    with rasterio.open(save_path) as src:
        transform = src.transform
        labelme_json = {}
        labelme_json['version'] = "4.5.7"
        labelme_json['flags'] =  {}
        # labelme_json['shapes'] = {}

        objects = []
        # convert to pixel coordinates
        for poly, class_int in zip(adj_polygons, adj_polygons_class):
            # print('polygon:', poly, class_int)
            x, y = [], []
            if isinstance(poly, Polygon):
                x, y = poly.exterior.coords.xy
            elif isinstance(poly, MultiPolygon):
                for sub_poly in poly.geoms:  # Iterate over individual Polygons
                    x_s, y_s = sub_poly.exterior.coords.xy
                    x.extend(x_s)
                    y.extend(y_s)
            pixel_xs, pixel_ys = raster_io.geo_xy_to_pixel_xy(x,y,transform)
            # print(pixel_xs,pixel_ys)

            points = [ [int(xx),int(yy)] for xx,yy in zip(pixel_xs,  pixel_ys) ]
            object_name = class_names[class_int]
            object = {'label':object_name}
            object['points'] = points

            objects.append(object)

        labelme_json['shapes'] = objects
        labelme_json['imageData'] = None
        labelme_json['imageHeight'] = src.height
        labelme_json['imageWidth'] = src.width
        labelme_json['imagePath'] = os.path.basename(save_path)

        io_function.save_dict_to_txt_json(save_josn_path,labelme_json)

    return save_path, save_josn_path




def get_sub_images_and_json_files(polygons_shp, class_names,bufferSize, image_tile_list,
                              saved_dir, pre_name, dstnodata, out_format, brectangle=True, proc_num=1,image_equal_size=None):
    '''
    get sub-images and corresponding josn file (labelme)
    :param polygons_shp:
    :param bufferSize:
    :param image_tile_list:
    :param saved_dir:
    :param pre_name:
    :param dstnodata:
    :param brectangle:
    :param proc_num:
    :return:
    '''

    # read polygons
    t_shapefile = gpd.read_file(polygons_shp)
    center_polygons = t_shapefile.geometry.values
    if 'class_int' in t_shapefile.keys():
        class_labels = t_shapefile['class_int'].tolist()
    else:
        class_labels = [0]*len(center_polygons)

    if image_equal_size is not None:
        center_polygons = get_subImages.get_polygon_extent_same_size(center_polygons,image_equal_size)

    polygons_all = center_polygons
    class_labels_all = class_labels
    img_tile_boxes = get_subImages.get_image_tile_bound_boxes(image_tile_list)

    proc_num = min(1, proc_num) # for test, current version.
    extension = raster_io.get_file_extension(out_format)
    # go through each polygon
    if proc_num == 1:
        for idx, (c_polygon, c_class_int) in enumerate(zip(center_polygons,class_labels)):
            save_path = os.path.join(saved_dir,pre_name + f'_sub_{idx}{extension}')
            tif_path, json_path = get_one_sub_image_json_file(idx, c_polygon, c_class_int, class_names, image_tile_list, img_tile_boxes, save_path,
                                dstnodata, brectangle, bufferSize, polygons_all, class_labels_all, out_format)

    elif proc_num > 1:
        pass

        # parameters_list = [
        #     (idx, c_polygon, c_class_int, class_names, image_tile_list, img_tile_boxes, save_path,
        #                         dstnodata, brectangle, bufferSize, polygons_all, class_labels_all)
        #     for idx, (c_polygon, c_class_int) in enumerate(zip(center_polygons,class_labels))]
        # theadPool = Pool(proc_num)  # multi processes
        # results = theadPool.starmap(get_one_sub_image_json_file, parameters_list)  # need python3
    else:
        raise ValueError('Wrong process number: %s'%(proc_num))



def get_sub_images_pixel_json_files(polygons_shp,image_folder_or_path,image_pattern,class_names, bufferSize,dstnodata,
                                    saved_dir,b_rectangle,process_num, out_format,image_equal_size=None):

    # check training polygons
    assert io_function.is_file_exist(polygons_shp)

    # get image tile list
    # image_tile_list = io_function.get_file_list_by_ext(options.image_ext, image_folder, bsub_folder=False)
    if os.path.isdir(image_folder_or_path):
        image_tile_list = io_function.get_file_list_by_pattern(image_folder_or_path, image_pattern)
    else:
        assert io_function.is_file_exist(image_folder_or_path)
        image_tile_list = [image_folder_or_path]

    if len(image_tile_list) < 1:
        raise IOError('error, failed to get image tiles in folder %s' % image_folder_or_path)

    get_subImages.check_projection_rasters(image_tile_list)  # it will raise errors if found problems

    get_subImages.check_1or3band_8bit(image_tile_list)  # it will raise errors if found problems

    # need to check: the shape file and raster should have the same projection.
    if get_subImages.get_projection_proj4(polygons_shp) != get_subImages.get_projection_proj4(image_tile_list[0]):
        raise ValueError('error, the input raster (e.g., %s) and vector (%s) files don\'t have the same projection' % (
        image_tile_list[0], polygons_shp))

    # check these are EPSG:4326 projection
    if get_subImages.get_projection_proj4(polygons_shp).strip() == '+proj=longlat +datum=WGS84 +no_defs':
        bufferSize = get_subImages.meters_to_degress_onEarth(bufferSize)

    pre_name = os.path.splitext(os.path.basename(image_tile_list[0]))[0]

    saved_dir = os.path.join(saved_dir, pre_name+ '_subImages')
    if os.path.isdir(saved_dir) is False:
        io_function.mkdir(saved_dir)

    get_sub_images_and_json_files(polygons_shp,class_names, bufferSize, image_tile_list,
                              saved_dir, pre_name, dstnodata, out_format, brectangle=b_rectangle, proc_num=process_num,
                                  image_equal_size=image_equal_size)


def test_get_sub_images_pixel_json_files():
    print('\n')
    print('running test_get_sub_images_pixel_json_files')
    dir = os.path.expanduser('~/Data/Arctic/alaska/autoMapping/alaskaNS_yolov4_3/multi_inf_results')

    polygons_shp = os.path.join(dir, 'I0/I0_alaska_north_slope_planet_rgb_2020_alaskaNS_yolov4_3_exp3.shp')
    image_folder_or_path = io_function.read_list_from_txt(os.path.join(dir, '0.txt'))[0]
    process_num = 1
    bufferSize = 300
    dstnodata = 0
    saved_dir = './'
    b_rectangle = True
    image_pattern = '.tif'
    class_names = ['rts']

    get_sub_images_pixel_json_files(polygons_shp, image_folder_or_path, image_pattern,class_names, bufferSize, dstnodata, saved_dir,b_rectangle, process_num)


def get_sub_images_from_prediction_results(para_file,polygons_shp,image_folder_or_path,image_pattern,saved_dir,out_format,
                                           image_equal_size=None):

    class_names = parameters.get_string_list_parameters(para_file,'object_names')

    dstnodata = parameters.get_digit_parameters(para_file, 'dst_nodata', 'int')
    bufferSize = parameters.get_digit_parameters(para_file, 'buffer_size', 'int')
    rectangle_ext = parameters.get_string_parameters_None_if_absence(para_file, 'b_use_rectangle')
    if rectangle_ext is not None:
        b_rectangle = True
    else:
        b_rectangle = False

    process_num = parameters.get_digit_parameters(para_file,'process_num', 'int')

    get_sub_images_pixel_json_files(polygons_shp, image_folder_or_path, image_pattern, class_names, bufferSize,
                                    dstnodata, saved_dir, b_rectangle, process_num, out_format, image_equal_size=image_equal_size)

    pass


def main(options, args):

    polygons_shp = args[0]
    image_folder_or_path = args[1]  # folder for store image tile (many split block of a big image)
    image_pattern = options.image_pattern
    saved_dir = options.out_dir
    para_file = options.para_file
    out_format = options.out_format
    image_equal_size = options.image_equal_size

    if para_file is None:
        process_num = options.process_num
        bufferSize = options.bufferSize
        dstnodata = options.dstnodata
        b_rectangle = options.rectangle
        class_names = ['others','rts']

        get_sub_images_pixel_json_files(polygons_shp, image_folder_or_path, image_pattern,class_names, bufferSize,
                                        dstnodata, saved_dir,b_rectangle, process_num, out_format, image_equal_size=image_equal_size)
    else:
        polygons_shp = args[0]
        image_folder_or_path = args[1]  # folder for store image tile (many split block of a big image)

        get_sub_images_from_prediction_results(para_file,polygons_shp,image_folder_or_path,image_pattern,saved_dir,out_format,
                                               image_equal_size=image_equal_size)



if __name__ == '__main__':
    usage = "usage: %prog [options] polygons_shp image_folder_or_path"
    parser = OptionParser(usage=usage, version="1.0 2021-4-15")
    parser.description = 'Introduction: get sub Images from polygons, convert polygon to labelme format.'
    parser.add_option("-b", "--bufferSize",
                      action="store", dest="bufferSize", type=float,
                      help="buffer size is in the projection, normally, it is based on meters")
    parser.add_option("-e", "--image_pattern",
                      action="store", dest="image_pattern", default='*.tif',
                      help="the image pattern of the image file")
    parser.add_option("-o", "--out_dir",
                      action="store", dest="out_dir", default='./',
                      help="the folder path for saving output files")
    parser.add_option("-n", "--dstnodata", type=int,
                      action="store", dest="dstnodata", default=0,
                      help="the nodata in output images")
    parser.add_option("-r", "--rectangle",
                      action="store_true", dest="rectangle", default=False,
                      help="whether use the rectangular extent of the polygon")
    parser.add_option("", "--process_num", type=int,
                      action="store", dest="process_num", default=4,
                      help="the process number for parallel computing")

    parser.add_option("-p", "--para_file",
                      action="store", dest="para_file",
                      help="the parameters file")
    parser.add_option("-t", "--out_format",
                      action="store", dest="out_format",default='GTIFF',
                      help="the format of output images, GTIFF, PNG, JPEG, VRT, etc")
    parser.add_option("-s", "--image_equal_size", type=float,
                      action="store", dest="image_equal_size",
                      help="if set (in meters), then extract the centroid of each polygon, "
                           "buffer this value to a polygon,making each extracted image has the same width and height ")

    (options, args) = parser.parse_args()
    # print(options.no_label_image)
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)