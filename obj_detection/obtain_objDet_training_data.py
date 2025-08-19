#!/usr/bin/env python
# Filename: obtain_objDet_training_data.py 
"""
introduction: prepare training data for object detection, similar to "img_classification/get_organize_training_data.py"

"yolov4_dir/pre_yolo_data.py" only work for training data prepared from semantic segmetnation,

This file, will take images and vectors (polygon or bounding boxes) as input, then output for format good '
for object detection (e.g., YOLO)

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 19 August, 2025
"""
import os,sys
from optparse import OptionParser

import numpy as np

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters
import basic_src.basic as basic
import basic_src.io_function as io_function
import basic_src.timeTools as timeTools

import time
import random


def get_sub_images_multi_regions_for_training_YOLO(WORK_DIR,para_file):

    print("get and organize training data for object detection")

    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    SECONDS = time.time()

    training_regions = parameters.get_string_list_parameters_None_if_absence(para_file, 'training_regions')
    if training_regions is None or len(training_regions) < 1:
        raise ValueError('No training area is set in %s' % para_file)

    image_patch_labels_list_txts = []
    training_data_dir = os.path.join(WORK_DIR, 'training_data')
    if not os.path.isdir(training_data_dir):
        io_function.mkdir(training_data_dir)

    for area_idx, area_ini in enumerate(training_regions):
        basic.outputlogMessage(
            ' %d/%d: getting training data from region: %s' % (area_idx, len(training_regions), area_ini))
        area_name_remark_time = parameters.get_area_name_remark_time(area_ini)

        extract_img_dir = os.path.join(training_data_dir, area_name_remark_time)
        if os.path.isdir(extract_img_dir) is False:
            io_function.mkdir(extract_img_dir)
        area_data_type = parameters.get_string_parameters(area_ini, 'area_data_type')

        if area_data_type == 'image_patch':
            # directly read

            image_path_list, image_labels, patch_list_txt = read_sub_image_labels_one_region(extract_img_dir, para_file,
                                                                                             area_ini, b_training=True)
            image_patch_labels_list_txts.append(patch_list_txt)

        else:

            image_path_list, image_labels, patch_list_txt = \
                extract_sub_image_labels_one_region(extract_img_dir, para_file, area_ini, b_training=True,
                                                    b_convert_label=True)
            image_patch_labels_list_txts.append(patch_list_txt)

    expr_name = parameters.get_string_parameters(para_file, 'expr_name')

    # merge label list
    save_path = class_utils.get_merged_training_data_txt(training_data_dir, expr_name, len(training_regions))
    merge_imagePatch_labels_for_multi_regions(image_patch_labels_list_txts, save_path)

    a_few_shot_samp_count = parameters.get_digit_parameters_None_if_absence(para_file, 'a_few_shot_samp_count', 'int')
    b_sep_train_valid_set_by_grids = parameters.get_bool_parameters_None_if_absence(para_file,
                                                                                    'b_sep_train_valid_set_by_grids')
    if b_sep_train_valid_set_by_grids is None:
        b_sep_train_valid_set_by_grids = False
    b_a_few_shot_training = parameters.get_bool_parameters(para_file, 'a_few_shot_training')
    if b_a_few_shot_training and a_few_shot_samp_count is not None:
        # backup the original file
        save_path_all_samp = io_function.get_name_by_adding_tail(save_path, 'all')
        io_function.copy_file_to_dst(save_path, save_path_all_samp)
        randomly_select_k_samples_each_classes(save_path_all_samp, save_path, sample_count=a_few_shot_samp_count,
                                               b_sep_by_grid=b_sep_train_valid_set_by_grids)

    duration = time.time() - SECONDS
    os.system(
        'echo "$(date): time cost of getting training data (image classification): %.2f seconds">>time_cost.txt' % duration)

    pass

def main(options, args):

    para_file = args[0]
    WORK_DIR = os.getcwd()
    get_sub_images_multi_regions_for_training_YOLO(WORK_DIR,para_file)

if __name__ == '__main__':

    usage = "usage: %prog [options] para_file "
    parser = OptionParser(usage=usage, version="1.0 2024-04-26")
    parser.description = 'Introduction: extract sub-images and sub-labels '

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)