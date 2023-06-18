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

import parameters
code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import basic_src.io_function as io_function
import datasets.raster_io as raster_io


def extract_points_from_polygons(para_file):
    multi_inf_regions = parameters.get_string_list_parameters(para_file, 'inference_regions')
    for area_idx, area_ini in enumerate(multi_inf_regions):
        training_polygon_shp = parameters.get_file_path_parameters_None_if_absence(area_ini,'training_polygons')
        if training_polygon_shp is None:
            continue

        # get image resolution
        inf_image_dir = parameters.get_directory(area_ini, 'inf_image_dir')
        inf_image_or_pattern = parameters.get_string_parameters(area_ini, 'inf_image_or_pattern')
        inf_img_list = io_function.get_file_list_by_pattern(inf_image_dir, inf_image_or_pattern)
        img_count = len(inf_img_list)
        if img_count < 1:
            raise ValueError('No image for inference, please check inf_image_dir (%s) and inf_image_or_pattern (%s) in %s'
                             % (inf_image_dir, inf_image_or_pattern, area_ini))
        xres, yres = raster_io.get_xres_yres_file(inf_img_list[0])



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
    usage = "usage: %prog [options] para_file"
    parser = OptionParser(usage=usage, version="1.0 2023-06-17")
    parser.description = 'Introduction: convert training polygons to prompts (points or boxes) '


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)