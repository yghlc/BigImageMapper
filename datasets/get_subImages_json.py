#!/usr/bin/env python
# Filename: get_subImages_json 
"""
introduction: copy sub-images and create label images based on json files
For each sub-image, it may have a corresponding json files storing polygon (label) of each object but in pixels

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 13 April, 2021
"""

import sys,os
from optparse import OptionParser

# import these two to make sure load GEOS dll before using shapely
import shapely
from shapely.geometry import Polygon

import rasterio
from rasterio.mask import mask
from rasterio.features import rasterize
import numpy as np

codes_dir2 = os.path.expanduser('~/codes/PycharmProjects/DeeplabforRS')
sys.path.insert(0, codes_dir2)

import basic_src.io_function as io_function
import basic_src.basic as basic
import raster_io

def find_corresonding_json_file(tif_path):
    json_path = os.path.splitext(tif_path)[0] + '.json'
    if os.path.isfile(json_path):
        return json_path
    return None

def shape_points_2_polygons(points, rasterio_transform):
    points_geo = [ raster_io.pixel_xy_to_geo_xy(point[0], point[1], rasterio_transform) for point in points ]
    poly = Polygon(points_geo)
    return poly

def create_label_from_json_pixel(json_file, tif_path, class_names,save_path):
    data_dict = io_function.read_dict_from_txt_json(json_file)
    # polygons
    shapes = data_dict['shapes']
    class_int = [class_names.index(shape['label']) + 1 for shape in shapes]
    shape_points = [shape['points'] for shape in shapes]


    with rasterio.open(tif_path) as src:

        transform = src.transform

        # convert shape points to polygons.
        polygons = [shape_points_2_polygons(shape, transform) for shape in shape_points]

        burn_out = np.zeros((src.height, src.width))

        # rasterize the shapes
        burn_shapes = [(item_shape, item_class_int) for (item_shape, item_class_int) in
                       zip(polygons, class_int)]
        #
        out_label = rasterize(burn_shapes, out=burn_out, transform=transform,
                              fill=0, all_touched=False, dtype=rasterio.uint8)

        # test: save it to disk
        kwargs = src.meta
        kwargs.update(
            dtype=rasterio.uint8,
            count=1)
        #   width=burn_out.shape[1],
        #    height=burn_out.shape[0],#transform=transform
        # remove nodta in the output
        if 'nodata' in kwargs.keys():
            del kwargs['nodata']

        with rasterio.open(save_path, 'w', **kwargs) as dst:
            dst.write_band(1, out_label.astype(rasterio.uint8))



def get_subimages_label_josn(input_image_dir,file_pattern, subImage_dir, subLabel_dir, class_names, b_no_label_image=False, process_num=1):

    sub_images_list = io_function.get_file_list_by_pattern(input_image_dir, file_pattern)
    if len(sub_images_list) < 1:
        basic.outputlogMessage('No sub-images in: %s with pattern: %s'%(input_image_dir, file_pattern))
        return False

    # do we need to check the projection of each sub-images?

    if os.path.isdir(subLabel_dir) is False:
        io_function.mkdir(subLabel_dir)
    if os.path.isdir(subImage_dir) is False:
        io_function.mkdir(subImage_dir)

    label_path_list = []
    if b_no_label_image is True:
        pass
    else:
        # create label images
        json_file_list = [find_corresonding_json_file(sub_img) for sub_img in sub_images_list]
        for tif_path, json_file in zip(sub_images_list,json_file_list):
            if json_file is None:
                label_path_list.append(None)
                continue
            save_path = os.path.join(subLabel_dir, io_function.get_name_no_ext(json_file) + '.tif')
            create_label_from_json_pixel(json_file, tif_path, class_names, save_path)
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


def test_create_label_from_json_pixel():
    print('test create_label_from_json_pixel')
    data_dir = os.path.expanduser('~/Data/Arctic/alaska/time_series_sub_images_191Polys/hillshade_sub_images')
    tif_path = os.path.join(data_dir,'Alaska_north_slope_hillshade_poly_24_timeSeries/Alaska_north_slope_hillshade_20170220_poly_24.tif')
    json_file = find_corresonding_json_file(tif_path)
    class_names = ['rts']
    save_path = 'Alaska_north_slope_hillshade_20170220_poly_24_label.tif'

    create_label_from_json_pixel(json_file, tif_path, class_names, save_path)



if __name__ == '__main__':
    pass