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
import basic_src.basic as basic
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
    poly_ids = []
    p_id = 1
    for p_list, c_value in zip(points_2d, class_values):
        points_list.extend(p_list)
        point_classes.extend([c_value]*len(p_list))
        poly_ids.extend([p_id]*len(p_list))
        p_id += 1
    if len(points_list) < 1:
        basic.outputlogMessage('There is not points after sampling, please consider increasing max_point_count')
        return None
    # save to file
    wkt_string = map_projection.get_raster_or_vector_srs_info_proj4(polygons_path)
    point_df = pd.DataFrame({'id':[i+1 for i in range(len(points_list))],
                             'poly_id':poly_ids,
                               'points':points_list,
                               'class_int':point_classes})
    vector_gpd.save_points_to_file(point_df,'points',wkt_string,save_path)


def extract_points_from_polygons(area_ini, prompt_save_folder, max_points_from_polygon,b_representative=False):

    training_polygon_shp = parameters.get_file_path_parameters_None_if_absence(area_ini,'training_polygons')
    if training_polygon_shp is None:
        basic.outputlogMessage('training polygons is not set in %s'%os.path.abspath(area_ini))
        return None
    point_save_path = os.path.join(prompt_save_folder, os.path.basename(io_function.get_name_by_adding_tail(training_polygon_shp,'points')))
    if os.path.isfile(point_save_path):
        basic.outputlogMessage('%s already exists, skipping sampling points'%point_save_path)
        return point_save_path

    if b_representative:
        extract_representative_point_from_polygons(training_polygon_shp,point_save_path)
    else:
        sample_points_from_polygons(training_polygon_shp,point_save_path, max_points_from_polygon)
    return point_save_path

def polygon_to_boxes(polygons_path, save_path):
    polygons, class_values = vector_gpd.read_polygons_attributes_list(polygons_path, 'class_int')
    boxes = [ vector_gpd.convert_bounds_to_polygon(vector_gpd.get_polygon_bounding_box(poly)) for poly in polygons]
    # save to file
    wkt_string = map_projection.get_raster_or_vector_srs_info_proj4(polygons_path)
    id_list = [i + 1 for i in range(len(boxes))]
    box_df = pd.DataFrame({'id': id_list,
                           'poly_id': id_list,
                             'boxes': boxes,
                             'class_int': class_values})
    vector_gpd.save_points_to_file(box_df, 'boxes', wkt_string, save_path)

def extract_boxes_from_polygons(area_ini, prompt_save_folder):

    training_polygon_shp = parameters.get_file_path_parameters_None_if_absence(area_ini,'training_polygons')
    if training_polygon_shp is None:
        basic.outputlogMessage('training polygons is not set in %s' % os.path.abspath(area_ini))
        return None
    box_save_path = os.path.join(prompt_save_folder, os.path.basename(io_function.get_name_by_adding_tail(training_polygon_shp,'boxes')))

    if os.path.isfile(box_save_path):
        basic.outputlogMessage('%s already exists, skipping extracting boxes'%box_save_path)
        return box_save_path

    polygon_to_boxes(training_polygon_shp,box_save_path)
    return box_save_path

def extract_representative_point_from_polygons(polygons_path, save_path):
    polygons, class_values = vector_gpd.read_polygons_attributes_list(polygons_path, 'class_int')
    points = [vector_gpd.get_polygon_representative_point(poly) for poly in polygons]
    # save to file
    wkt_string = map_projection.get_raster_or_vector_srs_info_proj4(polygons_path)
    id_list = [i + 1 for i in range(len(points))]
    box_df = pd.DataFrame({'id': id_list,
                           'poly_id': id_list,
                           'points': points,
                           'class_int': class_values})
    vector_gpd.save_points_to_file(box_df, 'points', wkt_string, save_path)

def trainingPolygons_to_prompt_main(para_file):
    print("training Polygons (semantic segmentation) to Prompts (points or boxes)")
    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    prompt_type = parameters.get_string_parameters_None_if_absence(para_file,'prompt_type')
    if prompt_type is None:
        basic.outputlogMessage('prompt_type is not set, skipping getting prompts')
        return
    multi_inf_regions = parameters.get_string_list_parameters(para_file, 'inference_regions')

    prompt_save_folder = parameters.get_string_parameters(para_file, 'prompt_save_folder')
    prompt_save_folder = os.path.abspath(prompt_save_folder)
    if os.path.isdir(prompt_save_folder) is False:
        io_function.mkdir(prompt_save_folder)
    max_points_from_polygon = parameters.get_digit_parameters_None_if_absence(para_file,'max_points_from_polygon', 'int')
    b_representative_point = parameters.get_bool_parameters_None_if_absence(para_file,'b_representative_point')
    if b_representative_point is None:
        b_representative_point = False

    SECONDS = time.time()

    for area_idx, area_ini in enumerate(multi_inf_regions):
        prompt_path = parameters.get_file_path_parameters_None_if_absence(area_ini, 'prompt_path')
        if prompt_path is not None and os.path.isfile(prompt_path):
            basic.outputlogMessage('Prompt is set in %s and exists, no need to generate a new one'%os.path.abspath(area_ini))
            continue
        if prompt_type.lower() == 'point':
            prompt_save_path = extract_points_from_polygons(area_ini, prompt_save_folder, max_points_from_polygon,
                                                            b_representative=b_representative_point)
        elif prompt_type.lower() == 'box':
            prompt_save_path = extract_boxes_from_polygons(area_ini, prompt_save_folder)
        else:
            raise ValueError('Unknown prompt type: %s, only support point and box'%str(prompt_type))

        # modify area_ini and write prompt_save_path (relative path)
        if prompt_save_path is not None:
            parameters.write_Parameters_file(area_ini,'prompt_path', os.path.relpath(prompt_save_path))


    duration = time.time() - SECONDS
    os.system('echo "$(date): time cost of converting to training polygont to promots: %.2f seconds">>time_cost.txt' % duration)



def main(options, args):
    para_file = args[0]
    trainingPolygons_to_prompt_main(para_file)

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