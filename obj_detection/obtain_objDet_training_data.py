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
import basic_src.map_projection as map_projection
import datasets.vector_gpd as vector_gpd

import time
import random

import objDet_utils

def merge_imagePatch_labels_for_multi_regions(image_patch_labels_list_txts, save_path):
    with open(save_path, 'w') as outfile:
        for fname in image_patch_labels_list_txts:
            with open(fname) as infile:
                outfile.write(infile.read())


def extract_sub_images(train_grids_shp,image_dir, buffersize,image_or_pattern,extract_img_dir,dstnodata,process_num,rectangle_ext,b_keep_org_file_name):

    get_subImage_script = os.path.join(code_dir, 'datasets', 'get_subImages.py')

    command_string = get_subImage_script + ' -b ' + str(buffersize) + ' -e ' + image_or_pattern + \
                     ' -o ' + extract_img_dir + ' -n ' + str(dstnodata) + ' -p ' + str(process_num) \
                     + ' ' + rectangle_ext + ' --no_label_image '
    if b_keep_org_file_name:
        command_string += ' --b_keep_grid_name '
    command_string += train_grids_shp + ' ' + image_dir
    basic.os_system_exit_code(command_string)

def read_sub_image_boxes_one_region(extract_img_dir, para_file,area_ini, b_training=True):


    return None, None, None

def extract_sub_image_boxes_one_region(save_img_dir, para_file, area_ini, b_training=True):
    '''
     get some images and boxes (if available) from one region for object detection
    :param save_img_dir:  save directory for the saved images
    :param para_file: para_file
    :param area_ini: area file
    :param b_training: if True, will extract image for training, otherwise, will extract image for inference
    :return: image_path_list, boxes, patch_list_txt (file_path boxes_txt)
    '''

    # extract sub-images

    extract_img_dir = save_img_dir

    dstnodata = parameters.get_string_parameters(para_file, 'dst_nodata')
    if b_training:
        buffersize = parameters.get_string_parameters(para_file, 'train_buffer_size')
    else:
        buffersize = parameters.get_string_parameters(para_file, 'inf_buffer_size')
    rectangle_ext = parameters.get_string_parameters(para_file, 'b_use_rectangle')
    process_num = parameters.get_digit_parameters(para_file, 'process_num', 'int')


    b_keep_org_file_name = parameters.get_bool_parameters_None_if_absence(para_file, 'b_keep_org_file_name')
    if b_keep_org_file_name is None:
        b_keep_org_file_name = False

    area_name_remark_time = parameters.get_area_name_remark_time(area_ini)
    patch_list_txt = os.path.join(extract_img_dir, area_name_remark_time + '_patch_list.txt')

    if os.path.isfile(patch_list_txt):
        print('%s exists, read it directly' % patch_list_txt)
        image_path_labels = [item.split(':') for item in io_function.read_list_from_txt(patch_list_txt)]
        image_path_list = [item[0] for item in image_path_labels]
        image_boxes_txt_list = [item[1] for item in image_path_labels]
        return image_path_list, image_boxes_txt_list, patch_list_txt


    # for training
    image_dir = parameters.get_directory(area_ini, 'input_image_dir')  # train_image_dir
    image_or_pattern = parameters.get_string_parameters(area_ini,'input_image_or_pattern')  # train_image_or_pattern

    train_grids_shp = parameters.get_file_path_parameters(area_ini,'training_grids')
    train_polygon_box_shp = parameters.get_file_path_parameters(area_ini,'training_polygons_boxes')


    ## extract sub-images and the bounding boxes
    extract_done_indicator = os.path.join(extract_img_dir, 'extract_image_using_vector.done')
    if os.path.isfile(extract_done_indicator):
        basic.outputlogMessage('Warning, sub-images already been extracted, read them directly')
    else:
        extract_sub_images(train_grids_shp, image_dir, buffersize, image_or_pattern, extract_img_dir, dstnodata,
                           process_num, rectangle_ext, b_keep_org_file_name)

        if os.path.isfile(extract_done_indicator) is False:
            with open(extract_done_indicator, 'w') as f_obj:
                f_obj.writelines(
                    '%s image extracting, complete on %s \n' % (extract_img_dir, timeTools.get_now_time_str()))

    image_path_list = io_function.get_file_list_by_pattern(extract_img_dir, 'subImages/*.tif')

    if len(image_path_list) < 1:
        raise IOError(f'No sub-images in {extract_img_dir}/subImages')

    # check projection
    img_prj = map_projection.get_raster_or_vector_srs_info_proj4(image_path_list[0])
    grid_prj = map_projection.get_raster_or_vector_srs_info_proj4(train_grids_shp)
    polygon_box_proj = map_projection.get_raster_or_vector_srs_info_proj4(train_polygon_box_shp)
    if img_prj != grid_prj or img_prj != polygon_box_proj:
        raise ValueError(f'Map projection inconsistency between images, train_grids_shp, and train_polygon_box_shp for area: {area_ini},'
                         f'{img_prj}, \n {grid_prj}, \n{polygon_box_proj}')

    # get bounding boxes
    b_ignore_edge_objects = parameters.get_bool_parameters_None_if_absence(para_file,'b_ignore_edge_objects')
    if b_ignore_edge_objects is None:
        b_ignore_edge_objects = False
    boxes_txt_list = objDet_utils.get_bounding_boxes_from_vector_file(image_path_list, train_polygon_box_shp,
                                                        b_ignore_edge_objects=b_ignore_edge_objects,b_save_removal=True)


    # adding these polygons/boxes touch edges back
    # cancelled, this is not a good approach, when getting bounding boxes, for those touch edge but more than
    # half of the boxes/polgyons within the extent should be not removed, but keep there.


    # b_add_edge_objects_after_removal = parameters.get_bool_parameters_None_if_absence(para_file,'b_add_edge_objects_after_removal')
    # if b_add_edge_objects_after_removal is None:
    #     b_add_edge_objects_after_removal = False
    # if b_add_edge_objects_after_removal:
    #     removal_gpkg_list = io_function.get_file_list_by_pattern(extract_img_dir, 'subImages/*_removal.gpkg')
    #     if len(removal_gpkg_list) > 0:
    #         save_removal_merged = os.path.join(extract_img_dir,'removal_merge.gpkg')
    #         vector_gpd.merge_vector_files(removal_gpkg_list,save_removal_merged, format='GPKG')
    #         geometry_vector_gpd = vector_gpd.geometries_overlap_another_group(train_polygon_box_shp,save_removal_merged)
    #         train_polygon_box_edge_shp = os.path.join(extract_img_dir, 'train_polygon_box_touch_edge.gpkg')
    #         geometry_vector_gpd.to_file(train_polygon_box_edge_shp)

    #         ## to be completed? no, cancel approach, see above
    #         buffersize2 = 40?
    #         extract_img_dir2 = ?
    #         extract_sub_images(train_polygon_box_edge_shp, image_dir, buffersize2, image_or_pattern, extract_img_dir2, dstnodata,
    #                            process_num, rectangle_ext, b_keep_org_file_name)



    if os.path.isfile(patch_list_txt) is False:
        # save the relative path and label to file
        image_path_box_list = ['%s:%s' % (os.path.relpath(img), os.path.relpath(box_txt) ) for img, box_txt in zip(image_path_list, boxes_txt_list)]
        io_function.save_list_to_txt(patch_list_txt, image_path_box_list)


    return image_path_list, boxes_txt_list, patch_list_txt


def get_sub_images_multi_regions_for_training_YOLO(WORK_DIR,para_file):

    print("get and organize training data for object detection")

    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    SECONDS = time.time()

    training_regions = parameters.get_string_list_parameters_None_if_absence(para_file, 'training_regions')
    if training_regions is None or len(training_regions) < 1:
        raise ValueError('No training area is set in %s' % para_file)

    image_patch_boxes_list_txts = []
    training_data_dir = os.path.join(WORK_DIR, 'training_data')
    if not os.path.isdir(training_data_dir):
        io_function.mkdir(training_data_dir)

    for area_idx, area_ini in enumerate(training_regions):
        basic.outputlogMessage('%d/%d: getting training data from region: %s' % (area_idx+1, len(training_regions), area_ini))
        area_name_remark_time = parameters.get_area_name_remark_time(area_ini)

        extract_img_dir = os.path.join(training_data_dir, area_name_remark_time)
        if os.path.isdir(extract_img_dir) is False:
            io_function.mkdir(extract_img_dir)
        area_data_type = parameters.get_string_parameters(area_ini, 'area_data_type')

        if area_data_type == 'image_patch_boxes':
            # directly read
            image_path_list, image_labels, patch_list_txt = read_sub_image_boxes_one_region(extract_img_dir, para_file,
                                                                    area_ini, b_training=True)
            image_patch_boxes_list_txts.append(patch_list_txt)

        else:

            image_path_list, image_labels, patch_list_txt = \
                extract_sub_image_boxes_one_region(extract_img_dir, para_file, area_ini, b_training=True)
            image_patch_boxes_list_txts.append(patch_list_txt)

    expr_name = parameters.get_string_parameters(para_file, 'expr_name')

    # merge label list
    merged_image_boxes_path = objDet_utils.get_merged_training_data_txt(training_data_dir, expr_name, len(training_regions))
    merge_imagePatch_labels_for_multi_regions(image_patch_boxes_list_txts, merged_image_boxes_path)

    # split to tran and val sets
    # split training and validation datasets
    from datasets.train_test_split import train_test_split_main
    training_data_per = parameters.get_digit_parameters_None_if_absence(para_file, 'training_data_per','float')
    train_sample_txt = parameters.get_string_parameters(para_file, 'training_sample_list_txt')
    val_sample_txt = parameters.get_string_parameters(para_file, 'validation_sample_list_txt')
    Do_shuffle = True
    train_sample_txt, val_sample_txt =\
        train_test_split_main(merged_image_boxes_path, training_data_per, Do_shuffle, train_sample_txt, val_sample_txt)


    # save to YOLO (darknet) format
    objDet_utils.save_training_data_to_yolo_format_darknet(para_file,train_sample_txt,val_sample_txt)


    duration = time.time() - SECONDS
    os.system('echo "$(date): time cost of getting training data (image classification): %.2f seconds">>time_cost.txt' % duration)


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