#!/usr/bin/env python
# Filename: trainingPolys_to_prompts.py 
"""
introduction: convert training polygons (original for semantic segmentation) to point or box prompts

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 17 June, 2023
"""

import os, sys
import time
from datetime import datetime
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters
import basic_src.io_function as io_function
import basic_src.map_projection as map_projection
# import datasets.raster_io as raster_io
import datasets.vector_gpd as vector_gpd
import pandas as pd


def test_sample_points_from_polygons():
    poly_path = os.path.expanduser('~/Data/Arctic/canada_arctic/Willow_River/training_polygons/WR_training_polygons_v4.shp')
    save_path = 'WR_training_polygons_v4_points.shp'
    sample_points_from_polygons(poly_path,save_path,max_point_each_poly=10)

def sample_points_from_polygons(polygons_path, save_path, max_point_each_poly=10):
    '''
    sample some points within each polygons
    :param polygons_path: polygon path
    :param save_path: save path for polygons
    :param max_point_each_poly:
    :return:
    '''
    polygons, class_values = vector_gpd.read_polygons_attributes_list(polygons_path,'class_int')
    # print(polygons)
    # print(class_values)
    points_2d = [vector_gpd.sample_points_within_polygon(item,max_point_count=max_point_each_poly) for item in polygons]
    points_list = []
    point_classes = []
    for p_list, c_value in zip(points_2d, class_values):
        points_list.extend(p_list)
        point_classes.extend([c_value]*len(p_list))

    # save to file
    wkt_string = map_projection.get_raster_or_vector_srs_info_proj4(polygons_path)
    point_df = pd.DataFrame({'id':[i+1 for i in range(len(points_list))],
                               'points':points_list,
                               'class_int':point_classes})
    vector_gpd.save_points_to_file(point_df,'points',wkt_string,save_path)


def extract_points_from_polygons(para_file):
    multi_inf_regions = parameters.get_string_list_parameters(para_file, 'inference_regions')
    for area_idx, area_ini in enumerate(multi_inf_regions):
        training_polygon_shp = parameters.get_file_path_parameters_None_if_absence(area_ini,'training_polygons')
        if training_polygon_shp is None:
            continue

        # get image resolution
        # inf_image_dir = parameters.get_directory(area_ini, 'inf_image_dir')
        # inf_image_or_pattern = parameters.get_string_parameters(area_ini, 'inf_image_or_pattern')
        # inf_img_list = io_function.get_file_list_by_pattern(inf_image_dir, inf_image_or_pattern)
        # img_count = len(inf_img_list)
        # if img_count < 1:
        #     raise ValueError('No image for inference, please check inf_image_dir (%s) and inf_image_or_pattern (%s) in %s'
        #                      % (inf_image_dir, inf_image_or_pattern, area_ini))
        # xres, yres = raster_io.get_xres_yres_file(inf_img_list[0])

        max_points_from_polygon = parameters.get_digit_parameters_None_if_absence(para_file,'max_points_from_polygon','int')
        prompt_save_folder = parameters.get_string_parameters(para_file,'prompt_save_folder')
        prompt_save_folder = os.path.abspath(prompt_save_folder)
        if os.path.isdir(prompt_save_folder) is False:
            io_function.mkdir(prompt_save_folder)
        point_save_path = os.path.join(prompt_save_folder, os.path.basename(io_function.get_name_by_adding_tail(training_polygon_shp,'points')))

        sample_points_from_polygons(training_polygon_shp,point_save_path, max_points_from_polygon)



def trainingPolygons_to_promot_main(para_file):
    print("training Polygons (semantic segmentation) to Prompts (points or boxes)")
    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    prompt_type = parameters.get_string_parameters(para_file,'prompt_type')
    SECONDS = time.time()

    if prompt_type.lower() == 'point':
        extract_points_from_polygons(para_file)
    elif prompt_type.lower() == 'box':
        pass
    else:
        raise ValueError('Unknown prompt type: %s, only support point and box'%str(prompt_type))

    duration = time.time() - SECONDS
    os.system('echo "$(date): time cost of converting to yolo format: %.2f seconds">>time_cost.txt' % duration)



def main(options, args):
    para_file = args[0]
    trainingPolygons_to_promot_main(para_file)

if __name__ == '__main__':
    # test_sample_points_from_polygons()
    # sys.exit()
    usage = "usage: %prog [options] para_file"
    parser = OptionParser(usage=usage, version="1.0 2023-06-17")
    parser.description = 'Introduction: convert training polygons to prompts (points or boxes) '


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)