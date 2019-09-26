#!/usr/bin/env python
# Filename: get_subImages 
"""
introduction: get sub Images (and Labels) from training polygons directly, without gdal_rasterize

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 26 September, 2019
"""

import sys,os
from optparse import OptionParser

HOME = os.path.expanduser('~')
# path of DeeplabforRS
codes_dir2 = HOME + '/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)

import basic_src.io_function as io_function
import basic_src.basic as basic

import rasterio
from rasterio.mask import mask
import geopandas as gpd

def get_image_tile_bound_boxes(image_tile_list):
    '''
    get extent of all the images
    :param image_tile_list:  a list containing all the image path
    :return:  a list of boxes
    '''
    boxes = []
    for image_path in image_tile_list:
        with rasterio.open(image_path) as src:
            # the extent of the raster
            raster_bounds = src.bounds
            boxes.append(raster_bounds)

    return boxes


def get_adjacent_polygons(center_polygon, buffer_area):

    pass

def get_mask_image(selected_polygons, image_tile_list, image_tile_bounds ):
    '''
    get a mask image based on selected polygons, this image may cross two image tiles
    :param selected_polygons: selected polygons
    :param image_tile_list: image tiles.
    :return:
    '''

    # check it cross two or more images

    # if it will output a very large image (10000 by 10000 pixels), then raise a error

    pass

def get_sub_images_labels(t_polygons_shp, t_polygons_shp_all, bufferSize, saved_dir, dstnodata, brectangle = True):
    '''
    get sub images (and labels ) from training polygons
    :param t_polygons_shp: training polygon
    :param t_polygons_shp_all: the full set of training polygon, t_polygons_shp is a subset or equal to this one.
    :param bufferSize: buffer size of a center polygon to create a sub images
    :param saved_dir: output dir
    :param dstnodata: nodata when save for the output images
    :param brectangle: True: get the rectangle extent of a images.
    :return:
    '''



    # read polygons
    if t_polygons_shp_all == t_polygons_shp:
        t_shapefile_all = gpd.read_file(t_polygons_shp_all)

        class_labels = t_shapefile_all['class_int'].tolist()

        # read the polygons (in shapely format)
        center_polygons = t_shapefile_all.geometry.values  # list of shapely geometries
    else:

        t_shapefile = gpd.read_file(t_polygons_shp)
        class_labels = t_shapefile['class_int'].tolist()
        center_polygons = t_shapefile.geometry.values




    t_shapefile = gpd.read_file(t_polygons_shp)






    #extract the geometry in GeoJSON format

    if t_polygons_shp_all != t_polygons_shp:
        # find the training polygons in the full set
        pass


    # find the data in the shape


    pass

def main(options, args):

    t_polygons_shp = args[0]
    image_folder = args[1]   # folder for store image tile (many split block of a big image)

    # check training polygons
    assert io_function.is_file_exist(t_polygons_shp)
    t_polygons_shp_all = options.all_training_polygons
    if t_polygons_shp_all is None:
        basic.outputlogMessage('Warning, the full set of training polygons is not assigned, '
                               'it will consider the one in input argument is the full set of training polygons')
        t_polygons_shp_all = t_polygons_shp
    assert io_function.is_file_exist(t_polygons_shp_all)

    # get image tile list
    image_tile_list = io_function.get_file_list_by_ext(options.image_ext, image_folder, bsub_folder=False)
    if len(image_tile_list) < 1:
        raise IOError('error, failed to get image tiles in folder %s'%image_folder)

    #TODO:need to check: the shape file and raster should have the same projection.

    #
    bufferSize = options.bufferSize
    saved_dir = options.out_dir
    dstnodata = options.dstnodata
    get_sub_images_labels(t_polygons_shp, t_polygons_shp_all, bufferSize, saved_dir, dstnodata, brectangle=True)




if __name__ == "__main__":
    usage = "usage: %prog [options] training_polygons image_folder"
    parser = OptionParser(usage=usage, version="1.0 2019-9-26")
    parser.description = 'Introduction: get sub Images (and Labels) from training polygons directly, without gdal_rasterize. ' \
                         'The image and shape file should have the same projection.'
    parser.add_option("-f", "--all_training_polygons",
                      action="store", dest="all_training_polygons",
                      help="the full set of training polygons. If the one in the input argument "
                           "is a subset of training polygons, this one must be assigned")
    parser.add_option("-b", "--bufferSize",
                      action="store", dest="bufferSize",type=float,
                      help="buffer size is in the projection, normally, it is based on meters")
    parser.add_option("-e", "--image_ext",
                      action="store", dest="image_ext",default = '.tif',
                      help="the extension of the image file")
    parser.add_option("-o", "--out_dir",
                      action="store", dest="out_dir",
                      help="the folder path for saving output files")
    parser.add_option("-n", "--dstnodata",
                      action="store", dest="dstnodata",
                      help="the nodata in output images")
    parser.add_option("-r", "--rectangle",
                      action="store_true", dest="rectangle",default=False,
                      help="whether use the rectangular extent of the polygon")


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    # if options.para_file is None:
    #     basic.outputlogMessage('error, parameter file is required')
    #     sys.exit(2)

    main(options, args)