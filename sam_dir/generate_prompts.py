#!/usr/bin/env python
# Filename: generate_prompts.py 
"""
introduction: generate point prompts for Segment Anything from raster files such as DEM differences, NDWI

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 28 July, 2023
"""

import os, sys
import time
from datetime import datetime
from optparse import OptionParser

import numpy as np

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters
import basic_src.basic as basic
import basic_src.io_function as io_function
import datasets.raster_io as raster_io
import datasets.vector_gpd as vector_gpd

from trainingPolys_to_prompts import extract_representative_point_from_polygons, sample_points_from_polygons, polygon_to_boxes

import cv2
from skimage import measure

def merge_txts_into_one(txt_list, save_path=None):
    if isinstance(txt_list, list) is False:
        return txt_list
    if len(txt_list) == 1:
        return txt_list[0]
    tmp_list = []
    for txt in txt_list:
        # avoid duplication (when run multiple time)
        file_list = io_function.read_list_from_txt(txt)
        file_list_unique = [item for item in file_list if item not in tmp_list]

        tmp_list.extend(file_list_unique)
    out_path = txt_list[0] if save_path is None else save_path
    io_function.save_list_to_txt(out_path,tmp_list)
    return out_path


def merge_prompts_same_type_into_one(prompts_txt_list, save_path=None):
    # if not multiple (>1) files, return it directly
    if isinstance(prompts_txt_list, list) is False:
        return prompts_txt_list
    if len(prompts_txt_list) == 1:
        return prompts_txt_list[0]

    # need to use *.shp (not *.gpkg) because in "sam_predict.py", need to check file names end with .shp
    # prmopts are points or boxes, so, it should be easy to keep file size small than 2 GB
    print(save_path)
    prompt_point_path = save_path.replace('.txt', '_point.shp')
    prompt_box_path = save_path.replace('.txt', '_box.shp')
    all_vector_list = []
    for txt in prompts_txt_list:
        tmp_list = io_function.read_list_from_txt(txt)
        all_vector_list.extend( [ os.path.join(os.path.dirname(txt) ,item) for item in tmp_list ])

    prompt_point_txt_list = [item for item in all_vector_list if item.endswith('point.shp')]
    prompt_box_txt_list = [item for item in all_vector_list if item.endswith('box.shp')]

    vector_gpd.merge_vector_files(prompt_point_txt_list,prompt_point_path)
    vector_gpd.merge_vector_files(prompt_box_txt_list, prompt_box_path)

    io_function.save_list_to_txt(save_path,[os.path.basename(prompt_point_path), os.path.basename(prompt_box_path)])
    return save_path

def binary_thresholding(image_2d, threshold, b_greater=False, b_morphology=False, morp_k_size=3, min_area=10):
    '''
    thresholding
    :param image_2d: 2d array
    :param threshold:  threshold
    :param b_greater: if True, pixel greater than the threshold is 1, otherwise, pixel less than the threshold is 1
    :param b_morphology: applying morphology operation, opening, and close
    :param morp_k_size: the kernel size for morphology operation
    :param min_area: regions with area less than "min_area" will be removed
    :return:
    '''

    if image_2d.ndim !=2:
        raise ValueError('Only support 2D image array')

    # binary
    np_binary = np.zeros_like(image_2d).astype(np.uint8)
    if b_greater:
        np_binary[image_2d >= threshold] = 1
    else:
        np_binary[image_2d < threshold] = 1

    # post-processing
    if b_morphology:
        # Dilation or opening
        # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
        kernel = np.ones((morp_k_size, morp_k_size), np.uint8)  # if kernal is 5 or larger, will remove some narrow parts.
        # bin_slope = cv2.dilate(bin_slope,kernel,iterations = 1)
        np_binary = cv2.morphologyEx(np_binary, cv2.MORPH_OPEN, kernel)     # use opening to remove some noise
        np_binary = cv2.morphologyEx(np_binary, cv2.MORPH_CLOSE, kernel)    # closing small holes inside

    # edit the raster directly using skimage, remove some big or tiny regions?
    # # set background, label 0 will be ignored
    # labels = measure.label(np_binary, background=0, connectivity=2)  # 2-connectivity, 8 neighbours
    # # get region
    # regions = measure.regionprops(labels)
    #  to complete

    return np_binary


def extract_points_from_polygons(polygons_shp, point_save_path, b_representative=False, max_points_from_polygon=5):
    if b_representative:
        extract_representative_point_from_polygons(polygons_shp,point_save_path)
    else:
        sample_points_from_polygons(polygons_shp,point_save_path, max_points_from_polygon)


def extract_boxes_from_polygons(polygons_path, save_path):
    return polygon_to_boxes(polygons_path, save_path)

def get_prompts_from_one_dem_diff(demD_file, prompt_type, prompt_save_folder, max_points_one_region, b_representative=False,
                                  dem_diff_thread_m=-1.0):
    '''
    get prompts for elevation difference by thresholding
    :param dem_file:
    :param prompt_type:
    :param prompt_save_folder:
    :param max_points_one_region:
    :param b_representative:
    :return:
    '''

    prompt_save_path = os.path.join(prompt_save_folder, os.path.basename(
        io_function.get_name_no_ext(demD_file) + '_prompts_thr%s.txt'%str(dem_diff_thread_m)))
    if os.path.isfile(prompt_save_path):
        basic.outputlogMessage('%s already exists, skip getting prompts' % prompt_save_path)
        return prompt_save_path

    demD_height, demD_width, demD_band_num, demD_date_type = raster_io.get_height_width_bandnum_dtype(demD_file)

    # if int16, then it's in centimeter
    if demD_date_type == 'int16':
        dem_diff_thread_m = dem_diff_thread_m*100
    # binary demD file
    demD_np, demD_nodata = raster_io.read_raster_one_band_np(demD_file)
    np_binary = binary_thresholding(demD_np, dem_diff_thread_m, b_greater=False, b_morphology=True, morp_k_size=5)

    # save to file
    save_bin_path = os.path.join(prompt_save_folder, os.path.basename(io_function.get_name_by_adding_tail(demD_file,'bin')))
    raster_io.save_numpy_array_to_rasterfile(np_binary, save_bin_path, demD_file, nodata=0, compress='lzw', tiled='yes', bigtiff='if_safer')

    # convert to polygons
    if os.path.isfile(save_bin_path):
        io_function.delete_shape_file(save_bin_path)
    bin_shp_path = vector_gpd.raster2shapefile(save_bin_path, connect8=True)


    # post-processing based on the polygons
    polygons = vector_gpd.read_polygons_gpd(bin_shp_path,b_fix_invalid_polygon=False)
    vector_gpd.add_attributes_to_shp(bin_shp_path, {'id': [item + 1 for item in range(len(polygons))] ,'class_int': [1]*len(polygons),
                                                    'poly_area':[poly.area for poly in polygons]})

    # more post-processing based for polygons????
    # remove small and big polygons?

    prompt_point_path = prompt_save_path.replace('.txt', '_point.shp')
    prompt_box_path = prompt_save_path.replace('.txt', '_box.shp')
    if prompt_type.lower() == 'point':
        extract_points_from_polygons(bin_shp_path, prompt_point_path,b_representative=b_representative,
                                     max_points_from_polygon = max_points_one_region)
        io_function.save_list_to_txt(prompt_save_path,[os.path.basename(prompt_point_path)])
    elif prompt_type.lower() == 'box':
        extract_boxes_from_polygons(bin_shp_path, prompt_box_path)
        io_function.save_list_to_txt(prompt_save_path, [os.path.basename(prompt_box_path)])
    elif prompt_type.lower() == 'point+box':
        extract_points_from_polygons(bin_shp_path, prompt_point_path, b_representative=b_representative,
                                     max_points_from_polygon=max_points_one_region)
        extract_boxes_from_polygons(bin_shp_path, prompt_box_path)
        io_function.save_list_to_txt(prompt_save_path, [os.path.basename(prompt_point_path), os.path.basename(prompt_box_path)])
    else:
        raise ValueError('Unknown prompt type: %s, only support point and box' % str(prompt_type))

    return prompt_save_path

def extract_prompts_from_dem_diff(area_ini, prompt_type, prompt_save_folder, max_points_one_region, b_representative=False,
                                 dem_diff_thread_m=-1.0):

    # get files
    dem_diff_file_dir = parameters.get_directory(area_ini, 'dem_diff_prompt_dir')
    dem_diff_file_or_pattern = parameters.get_string_parameters(area_ini, 'dem_diff_prompt_or_pattern')
    dem_diff_file_list = io_function.get_file_list_by_pattern(dem_diff_file_dir, dem_diff_file_or_pattern)

    if len(dem_diff_file_list) < 1:
        raise IOError('No DEM Diff file found by \n dem_diff_file_dir: %s \n dem_diff_file_or_pattern: %s'%(dem_diff_file_dir, dem_diff_file_or_pattern))
    else:
        basic.outputlogMessage('find %d DEM diff files'%len(dem_diff_file_list))

    prompt_save_list = []

    for idx, dem_diff_file in enumerate(dem_diff_file_list):
        basic.outputlogMessage('%d/%d, getting prompts from a DEM diff file'%(idx+1, len(dem_diff_file_list)))
        prompt_save_path = get_prompts_from_one_dem_diff(dem_diff_file, prompt_type,prompt_save_folder,
                                                         max_points_one_region,b_representative=b_representative,
                                                         dem_diff_thread_m = dem_diff_thread_m)
        prompt_save_list.append(prompt_save_path)

    return prompt_save_list

def extract_prompts_from_raster(area_ini, para_file, prompt_save_folder, max_points_one_region,b_representative=False):
    '''
    extract prompts from raster for Segment Anything
    :param area_ini:
    :param para_file:
    :param prompt_save_folder:
    :param max_points_one_region:
    :param b_representative:
    :return:
    '''
    prompt_type = parameters.get_string_parameters_None_if_absence(para_file, 'prompt_type')

    if prompt_type is None:
        basic.outputlogMessage('prompt_type is not set, skipping getting prompts')
        return

    prompt_source_data = parameters.get_string_parameters(para_file, 'prompt_source_data')
    if prompt_source_data.lower() == 'dem_diff':
        dem_diff_thread_m = parameters.get_digit_parameters(para_file, 'dem_diff_threshold_m', 'float')
        prompt_save_path = extract_prompts_from_dem_diff(area_ini, prompt_type, prompt_save_folder, max_points_one_region,
                                                        b_representative=b_representative, dem_diff_thread_m=dem_diff_thread_m)
    elif prompt_source_data.lower() == 'ndwi':
        raise ValueError('not support yet')
    elif prompt_source_data.lower() == 'polygons':
        raise ValueError('not support yet, prompt_source_data: %s, try to use "trainingPolys_to_prompts.py"' % str(prompt_source_data))
    else:
        raise ValueError('not support yet, prompt_source_data: %s' % str(prompt_source_data))

    return prompt_save_path

def generate_prompts_main(para_file):
    print("raster files to Prompts (points or boxes)")
    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    multi_inf_regions = parameters.get_string_list_parameters(para_file, 'inference_regions')

    prompt_save_folder = parameters.get_string_parameters(para_file, 'prompt_save_folder')
    prompt_save_folder = os.path.abspath(prompt_save_folder)
    if os.path.isdir(prompt_save_folder) is False:
        io_function.mkdir(prompt_save_folder)

    # max_points_from_polygon: max point for each separate region
    max_points_from_polygon = parameters.get_digit_parameters_None_if_absence(para_file, 'max_points_from_polygon','int')
    b_representative_point = parameters.get_bool_parameters_None_if_absence(para_file, 'b_representative_point')
    if b_representative_point is None:
        b_representative_point = False

    SECONDS = time.time()

    for area_idx, area_ini in enumerate(multi_inf_regions):
        prompt_path = parameters.get_file_path_parameters_None_if_absence(area_ini, 'prompt_path')
        if prompt_path is not None and os.path.isfile(prompt_path):
            basic.outputlogMessage('Prompt is set in %s and exists, no need to generate a new one' % os.path.abspath(area_ini))
            continue

        area_name = parameters.get_string_parameters(area_ini, 'area_name')
        area_remark = parameters.get_string_parameters(area_ini, 'area_remark')
        area_time = parameters.get_string_parameters(area_ini, 'area_time')
        area_name_remark_time = area_name + '_' + area_remark + '_' + area_time

        prompt_save_path = extract_prompts_from_raster(area_ini, para_file, prompt_save_folder, max_points_from_polygon,
                                                        b_representative=b_representative_point)

        merged_txt = os.path.join(prompt_save_folder, area_name_remark_time + "_prompts.txt")
        # prompt_save_path = merge_txts_into_one(prompt_save_path,save_path=merged_txt)
        prompt_save_path = merge_prompts_same_type_into_one(prompt_save_path,merged_txt)

        # modify area_ini and write prompt_save_path (relative path)
        if prompt_save_path is not None and isinstance(prompt_save_path, list) is False:
            parameters.write_Parameters_file(area_ini, 'prompt_path', os.path.relpath(prompt_save_path))

    duration = time.time() - SECONDS
    os.system(
        'echo "$(date): time cost of converting to training polygons to prompts: %.2f seconds">>time_cost.txt' % duration)

def main(options, args):
    para_file = args[0]
    generate_prompts_main(para_file)


if __name__ == '__main__':
    usage = "usage: %prog [options] para_file"
    parser = OptionParser(usage=usage, version="1.0 2023-07-28")
    parser.description = 'Introduction: generate prompts (points or boxes) from raster files'

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)

