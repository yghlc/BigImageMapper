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

import basic_src.basic as basic
import basic_src.io_function as io_function

import rasterio
import geopandas as gpd
import numpy as np
from rasterio.features import rasterize

sys.path.insert(0, os.path.join(code_dir,'datasets'))
from get_subImages import get_projection_proj4

def find_corresponding_geojson_SpaceNet(tif_path, geojson_list, geojson_name_list):
    img_name = os.path.basename(tif_path)
    temp_list = os.path.splitext(img_name)[0].split('_')[1:]
    com_name = '_'.join(temp_list)
    # find string like: "AOI_2_Vegas_img4491"
    for g_name, g_path in zip(geojson_name_list,geojson_list):
        if com_name in g_name:
            return g_path

    return None


def get_subimages_SpaceNet(input_image_dir,image_pattern,input_polygon_dir, polygon_pattern, subImage_dir,subLabel_dir,process_num=1,
                           burn_value=1, b_no_label_image=False):

    sub_images_list = io_function.get_file_list_by_pattern(input_image_dir, image_pattern)
    if len(sub_images_list) < 1:
        basic.outputlogMessage('No sub-images in: %s with pattern: %s'%(input_image_dir, image_pattern))
        return False

    sub_images_count = len(sub_images_list)
    # do we need to check the projection of each sub-images?

    if os.path.isdir(subLabel_dir) is False:
        io_function.mkdir(subLabel_dir)
    if os.path.isdir(subImage_dir) is False:
        io_function.mkdir(subImage_dir)

    label_path_list = []
    if b_no_label_image is True:
        pass
    else:
        # polygon file list
        polygon_files_list = io_function.get_file_list_by_pattern(input_polygon_dir, polygon_pattern)
        if len(polygon_files_list) < 1:
            basic.outputlogMessage('No polygon files in: %s with pattern: %s' % (input_polygon_dir, polygon_pattern))
            return False

        polygon_name_list = [os.path.basename(item) for item in polygon_files_list ]

        # create label images
        for idx, tif_path in enumerate(sub_images_list):
            print('%d / %d create label raster for %s'%(idx,sub_images_count,tif_path))
            # find polygon file
            poly_path = find_corresponding_geojson_SpaceNet(tif_path,polygon_files_list, polygon_name_list)
            if poly_path is None:
                print('Warning, cannot find corresponding polygon files')
                continue

            save_path = os.path.join(subLabel_dir, io_function.get_name_no_ext(poly_path) + '.tif')
            if os.path.isfile(save_path):
                print('warning, %s already exists, skip'%save_path)
                label_path_list.append(save_path)
                continue
            if rasterize_polygons_to_ref_raster(tif_path, poly_path, burn_value, None, save_path,
                                             datatype='Byte', ignore_edge=True) is True:
                label_path_list.append(save_path)

    # copy sub-images, adding to txt files
    with open('sub_images_labels_list.txt','a') as f_obj:
        for tif_path, label_file in zip(sub_images_list, label_path_list):
            if label_file is None:
                continue
            dst_subImg = os.path.join(subImage_dir, os.path.basename(tif_path))

            # copy sub-images
            io_function.copy_file_to_dst(tif_path,dst_subImg, overwrite=True)

            sub_image_label_str = dst_subImg + ":" + label_file + '\n'
            f_obj.writelines(sub_image_label_str)

    return True




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

    if isinstance(burn_value,int) is False or isinstance(burn_value,float):
        raise ValueError('The burn value should be int or float')

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
        out_label = burn_out

        if len(polygons) > 0:
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
                # after negative buffer, some small polygons may be deleted, need to check
                inner_list = [ item for item in inner_list if item.is_valid and item.is_empty is False]
                if len(inner_list) > 0:
                    # rasterize the outer
                    burn_shapes = [(item_shape, 255) for item_shape in outer_list]
                    out_label = rasterize(burn_shapes, out=burn_out, transform=transform, fill=0, all_touched=False, dtype=dtype)
                    # rasterize the inner  parts
                    burn_shapes = [(item_shape, item_class_int) for (item_shape, item_class_int) in zip(inner_list, class_labels)]
                    out_label = rasterize(burn_shapes, out=out_label, transform=transform, fill=0, all_touched=False, dtype=dtype)
                else:
                    print('After negative buffer operation, there is no polygon valid for rasterizing')
        else:
            print('Warning, no Polygon in %s, will save a dark label'%poly_path)

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
    if os.path.isfile(save_path):
        print('Warning, %s already exists'%save_path)
        return True

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
                      action="store", dest="burn_value",default=1, type=int,
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
