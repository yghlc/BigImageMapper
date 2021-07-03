#!/usr/bin/env python
# Filename: rasterize_polygons 
"""
introduction: rasterize polgyons into raster using rasterio

difference from get_sub_label in "get_subImages.py", this script rasterizes the entire scene, not only the surrounding of a polygon

A similar scrip is  ../../DeeplabforRS/prepare_raster.py with gdal_rasterize, but this one is more flexible.

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 02 July, 2021
"""

import sys,os
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)

import basic_src.io_function as io_function

import rasterio
import geopandas as gpd
import numpy as np
from rasterio.features import rasterize

sys.path.insert(0, os.path.join(code_dir,'datasets'))
from get_subImages import get_projection_proj4

def rasterize_polygons_to_ref_raster(ref_raster, poly_path, burn_value, attribute_name, save_path, datatype='Byte',ignore_edge=True):
    '''
    rasterize polygons to a raster with the same size of ref_raster
    :param ref_raster:  reference raster
    :param poly_path: shapefile path
    :param burn_value: if attribute_name is None, then we will use burn_value
    :param attribute_name: the attribute to be burned into raster
    :param save_path:
    :param datatype: datatype
    :param ignore_edge: if True, will burn a buffer areas (~5 pixel as 255)
    :return:
    '''

    # need to check the projection
    # need to check: the shape file and raster should have the same projection.
    if get_projection_proj4(ref_raster) != get_projection_proj4(poly_path):
        raise ValueError('error, the input raster (e.g., %s) and vector (%s) files don\'t have the same projection'%(ref_raster, poly_path))

    # read polygons (can read geojson file directly)
    shapefile = gpd.read_file(poly_path)
    polygons = shapefile.geometry.values
    if attribute_name is None:
        class_labels = [burn_value] * len(polygons)
    else:
        class_labels = shapefile[attribute_name].tolist()

    # output datatype: https://gdal.org/programs/gdal_rasterize.html
    # -ot {Byte/Int16/UInt16/UInt32/Int32/Float32/Float64/
    #             CInt16/CInt32/CFloat32/CFloat64}]
    if datatype == 'Byte':
        dtype = rasterio.uint8
    elif datatype == 'UInt16':
        dtype = rasterio.uint16
    else:
        dtype = rasterio.int32

    with rasterio.open(ref_raster) as src:

        transform = src.transform
        burn_out = np.zeros((src.height, src.width))

        if ignore_edge is False:
            # rasterize the shapes
            burn_shapes = [(item_shape, item_class_int) for (item_shape, item_class_int) in
                           zip(polygons, class_labels)]
            #
            out_label = rasterize(burn_shapes, out=burn_out, transform=transform,
                                  fill=0, all_touched=False, dtype=dtype)
        else:
            # burn a buffer area (4 to 5 pixel) of edge as 255
            xres, yres = src.res
            outer_list = [poly.buffer(2 *xres) for poly in polygons]
            inner_list = [poly.buffer(-2 *xres) for poly in polygons]

            # rasterize the outer
            burn_shapes = [(item_shape, 255) for item_shape in outer_list]
            out_label = rasterize(burn_shapes, out=burn_out, transform=transform, fill=0, all_touched=False, dtype=dtype)
            # rasterize the inner  parts
            burn_shapes = [(item_shape, item_class_int) for (item_shape, item_class_int) in zip(inner_list, class_labels)]
            out_label = rasterize(burn_shapes, out=out_label, transform=transform, fill=0, all_touched=False, dtype=dtype)

        # save it to disk
        kwargs = src.meta
        kwargs.update(
            dtype=dtype,
            count=1)

        if 'nodata' in kwargs.keys():
            del kwargs['nodata']

        with rasterio.open(save_path, 'w', **kwargs) as dst:
            dst.write_band(1, out_label.astype(dtype))

    return True


def rasterize_polygons(poly_path, burn_value, attribute_name, xres,yres, save_path, datatype='Byte'):
    '''

    :param poly_path:
    :param burn_value:
    :param attribute_name:
    :param xres:
    :param yres:
    :param save_path:
    :param datatype:
    :return:
    '''

    # not finish
    # ref to:  ../../DeeplabforRS/prepare_raster.py

    # layername =  os.path.splitext(os.path.basename(shp_path))[0]
    # args_list = ['gdal_rasterize', '-a', class_int_field, '-ot', 'Byte', \
    #              '-tr',str(res),str(res),'-l',layername,shp_path,raster_path]
    # result = basic.exec_command_args_list_one_file(args_list,raster_path)
    # if os.path.getsize(result) < 1:
    #     return False
    pass


def test_rasterize_polygons_to_ref_raster():
    data_dir = os.path.expanduser('~/Data/tmp_data/building/SN2_buildings_train_AOI_5_Khartoum/AOI_5_Khartoum_Train')
    geojson = os.path.join(data_dir,'geojson/buildings/buildings_AOI_5_Khartoum_img484.geojson')
    ref_raster = os.path.join(data_dir,'RGB-PanSharpen/RGB-PanSharpen_AOI_5_Khartoum_img484.tif')

    burn_value =1
    attribute_name = None
    save_path = os.path.join(data_dir,'img484_label.tif')
    rasterize_polygons_to_ref_raster(ref_raster, geojson, burn_value, attribute_name, save_path, datatype='Byte')


def main(options, args):
    shp_path = args[0]
    io_function.is_file_exist(shp_path)

    ref_raster = options.reference_raster
    # nodata = options.nodata
    out_dir = options.out_dir
    attribute_name = options.attribute
    burn_value = options.burn_value
    b_burn_edge = options.burn_edge_255

    file_name = os.path.splitext(os.path.basename(shp_path))[0]
    save_path = os.path.join(out_dir, file_name + '_label.tif')

    if ref_raster is not None:
        rasterize_polygons_to_ref_raster(ref_raster, shp_path, burn_value, attribute_name, save_path,ignore_edge=b_burn_edge)
    else:
        xres = options.pixel_size_x
        yres = options.pixel_size_y
        rasterize_polygons(shp_path, burn_value, attribute_name, xres, yres, save_path)



if __name__ == "__main__":
    usage = "usage: %prog [options] polygons_path"
    parser = OptionParser(usage=usage, version="1.0 2021-7-2")
    parser.description = 'Introduction: rasterize the polygons in the entire scene,  without gdal_rasterize. ' \
                         'The image and shape file should have the same projection.'
    parser.add_option("-r", "--reference_raster",
                      action="store", dest="reference_raster",
                      help="a raster file as reference, should have have the same projection")
    parser.add_option("-o", "--out_dir",
                      action="store", dest="out_dir", default='./',
                      help="the folder path for saving output files")

    parser.add_option("-e", "--burn_edge_255",
                      action="store_true", dest="burn_edge_255",default=False,
                      help="if set, it will burn the edge 4-5 pixel as 255")


    parser.add_option("-a", "--attribute",
                      action="store", dest="attribute",
                      help="the attribute name in the vector files for rasterization")
    parser.add_option("-b", "--burn_value",
                      action="store", dest="burn_value",default=1,
                      help="the burn value will be used if attribute is not assigned")
    parser.add_option("-x", "--pixel_size_x",
                      action="store", dest="pixel_size_x",
                      help="the x resolution of output raster, it will be ignored if reference_raster is set")
    parser.add_option("-y", "--pixel_size_y",
                      action="store", dest="pixel_size_y",
                      help="the y resolution of output raster, it will be ignored if reference_raster is set")


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)
