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

codes_dir2 = os.path.expanduser('~/codes/PycharmProjects/DeeplabforRS')
sys.path.insert(0, codes_dir2)

import basic_src.io_function as io_function
import basic_src.basic as basic

import geopandas as gpd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import get_subImages

# def get_sub_image(idx,selected_polygon, image_tile_list, image_tile_bounds, save_path, dstnodata, brectangle ):

def get_one_sub_image_json_file(idx, center_polygon, c_class_int, image_tile_list, image_tile_bounds, save_path, dstnodata, brectangle,
                                bufferSize, polygons_all, class_labels_all):
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
    get_subImages.get_sub_image(idx, expansion_polygon, image_tile_list,image_tile_bounds,save_path,dstnodata,brectangle)

    #
    save_josn_path = os.path.splitext(save_path)[0] + '.json'

    # get adjacent polygon
    adj_polygons, adj_polygons_class = get_subImages.get_adjacent_polygons(center_polygon, polygons_all, class_labels_all, bufferSize,
                                                             brectangle)
    # add the center polygons to adj_polygons
    adj_polygons.extend([center_polygon])
    adj_polygons_class.extend([c_class_int])

    # convert to pixel coordinates

    print(adj_polygons)
    print(adj_polygons_class)


    return save_path, save_josn_path




def get_sub_images_and_json_files(polygons_shp, bufferSize, image_tile_list,
                              saved_dir, pre_name, dstnodata, brectangle=True, proc_num=1):
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

    polygons_all = center_polygons
    class_labels_all = class_labels
    img_tile_boxes = get_subImages.get_image_tile_bound_boxes(image_tile_list)

    # go through each polygon
    if proc_num == 1:
        for idx, (c_polygon, c_class_int) in enumerate(zip(center_polygons,class_labels)):
            save_path = os.path.join(saved_dir,pre_name + '_sub_%d.tif'%idx)
            tif_path, json_path = get_one_sub_image_json_file(idx, c_polygon, c_class_int, image_tile_list, img_tile_boxes, save_path,
                                dstnodata, brectangle, bufferSize, polygons_all, class_labels_all)

    elif proc_num > 1:
        pass
        # parameters_list = [
        #     (idx,c_polygon, bufferSize,pre_name, pre_name_for_label,c_class_int,saved_dir, image_tile_list,
        #                     img_tile_boxes,dstnodata,brectangle, b_label,polygons_all,class_labels_all)
        #     for idx, (c_polygon, c_class_int) in enumerate(zip(center_polygons, class_labels))]
        # theadPool = Pool(proc_num)  # multi processes
        # results = theadPool.starmap(get_one_sub_image_label_parallel, parameters_list)  # need python3
    else:
        raise ValueError('Wrong process number: %s'%(proc_num))



def get_sub_images_pixel_json_files(polygons_shp,image_folder_or_path,bufferSize,dstnodata,saved_dir,b_rectangle,process_num):

    # check training polygons
    assert io_function.is_file_exist(polygons_shp)

    # get image tile list
    # image_tile_list = io_function.get_file_list_by_ext(options.image_ext, image_folder, bsub_folder=False)
    if os.path.isdir(image_folder_or_path):
        image_tile_list = io_function.get_file_list_by_pattern(image_folder_or_path, options.image_ext)
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

    get_sub_images_and_json_files(polygons_shp, bufferSize, image_tile_list,
                              saved_dir, pre_name, dstnodata, brectangle=b_rectangle, proc_num=process_num)


def test_get_sub_images_pixel_json_files():
    print('\n')
    print('running test_get_sub_images_pixel_json_files')
    dir = os.path.expanduser(
        '~/Data/Arctic/alaska/autoMapping/alaskaNS_yolov4_1/multi_inf_results/alaska_north_slope_hillshade_2010to2017')

    polygons_shp = os.path.join(dir, 'I0/I0_alaska_north_slope_hillshade_2010to2017_alaskaNS_yolov4_1_exp1.shp')
    image_folder_or_path = io_function.read_list_from_txt(os.path.join(dir, '0.txt'))[0]
    process_num = 1
    bufferSize = 300
    dstnodata = 0
    saved_dir = './'
    b_rectangle = True

    get_sub_images_pixel_json_files(polygons_shp, image_folder_or_path, bufferSize, dstnodata, saved_dir,b_rectangle, process_num)

def main(options, args):

    polygons_shp = args[0]
    image_folder_or_path = args[1]  # folder for store image tile (many split block of a big image)
    process_num = options.process_num
    bufferSize = options.bufferSize
    dstnodata = options.dstnodata
    saved_dir = options.out_dir
    b_rectangle = options.rectangle

    get_sub_images_pixel_json_files(polygons_shp, image_folder_or_path, bufferSize, dstnodata, saved_dir,b_rectangle, process_num)




if __name__ == '__main__':
    usage = "usage: %prog [options] polygons_shp image_folder_or_path"
    parser = OptionParser(usage=usage, version="1.0 2021-4-15")
    parser.description = 'Introduction: get sub Images from polygons, convert polygon to labelme format.'
    parser.add_option("-b", "--bufferSize",
                      action="store", dest="bufferSize", type=float,
                      help="buffer size is in the projection, normally, it is based on meters")
    parser.add_option("-e", "--image_ext",
                      action="store", dest="image_ext", default='*.tif',
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
    parser.add_option("-p", "--process_num", type=int,
                      action="store", dest="process_num", default=4,
                      help="the process number for parallel computing")

    (options, args) = parser.parse_args()
    # print(options.no_label_image)
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)