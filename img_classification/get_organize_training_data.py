#!/usr/bin/env python
# Filename: get_organize_training_data.py 
"""
introduction: prepare training data for image classification

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 26 April, 2024
"""

import io
import os,sys
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters
import basic_src.basic as basic
import basic_src.io_function as io_function
import basic_src.timeTools as timeTools

import time
import random

import class_utils

class_id_shp={'thawslump':1, 'background':0}
label_ids = {}

def convert_label_id_to_newSystem(image_labels,class_id_shp):

    ng_1_count_before = image_labels.count(-1)

    for key in class_id_shp.keys():
        original_id = class_id_shp[key]
        if key in label_ids.keys():
            new_id = label_ids[key]
        else:
            # set these into -1 if it is not in the new System
            new_id = -1
        if original_id != new_id:
            basic.outputlogMessage('warning, change class id (%d) to a new one (%d) for %s'%(original_id, new_id, key) )
            image_labels = [ new_id if ii==original_id else ii for ii in image_labels ]

    ng_1_count_after = image_labels.count(-1)
    if ng_1_count_after > ng_1_count_before:
        basic.outputlogMessage('Warning, After convert_label_id_to_newSystem, adding %d (-1) values (original count=%d)'
                               ''%(ng_1_count_after - ng_1_count_before, ng_1_count_before))

    if ng_1_count_after < ng_1_count_before:
        raise ValueError('ng_1_count_after smaller than ng_1_count_before, not expected')

    return image_labels


def read_label_ids(label_txt):
    global label_ids
    label_ids = read_label_ids_local(label_txt)

def read_label_ids_local(label_txt):
    # label_list = [[item.split(',')[0], int(item.split(',')[1])] for item in io_function.read_list_from_txt(label_txt)]
    label_ids = {}
    for item in io_function.read_list_from_txt(label_txt):
        tmp = item.split(',')
        label_ids[tmp[0].lower()] = int(tmp[1]) # change to lower case
    return label_ids


def randomly_select_k_samples_each_classes(image_labels_txt, save_path, sample_count=10):
    '''
    randonly select a few samples (k) for each class
    :param image_labels_txt:  all samples, each line contain: image_path class_id
    :param save_path:  save path
    :param sample_count: the count of samples want to select
    :return:
    '''
    image_path_list, image_labels = read_image_path_label_from_txt(image_labels_txt)
    # group classes
    class_id_images = {}
    for img, label in zip(image_path_list, image_labels):
        class_id_images.setdefault(label, []).append(img)

    sel_class_id_images = {}
    not_sel_class_id_images = {}
    # the number of sample for each class
    for key in class_id_images.keys():
        print('the sample count of class %d is %d'%(key, len(class_id_images[key])))
        if sample_count > len(class_id_images[key]):
            raise ValueError('for class %d, select sample count (%d) is larger than the total number (%d) of all samples'%(key, sample_count,len(class_id_images[key])))
        sel_class_id_images[key] = random.sample(class_id_images[key], sample_count)
        # save those not selected ones (no duplicated elements)
        not_sel_class_id_images[key] = list(set(class_id_images[key]) - set(sel_class_id_images[key]))

    # save to path
    image_path_list_sel = []
    image_labels_sel = []
    for key in sel_class_id_images.keys():
        image_path_list_sel.extend(sel_class_id_images[key])
        image_labels_sel.extend([key]*len(sel_class_id_images[key]))
    save_image_path_label_to_txt(image_path_list_sel, image_labels_sel, save_path)

    save_path_not_sel = io_function.get_name_by_adding_tail(save_path,'notSel')
    image_path_list_not_sel = []
    image_labels_not_sel = []
    for key in sel_class_id_images.keys():
        image_path_list_not_sel.extend(not_sel_class_id_images[key])
        image_labels_not_sel.extend([key]*len(not_sel_class_id_images[key]))
    save_image_path_label_to_txt(image_path_list_not_sel, image_labels_not_sel, save_path_not_sel)



def read_image_path_label_from_txt(patch_list_txt):
    image_path_labels = [item.split() for item in io_function.read_list_from_txt(patch_list_txt)]
    image_path_list = [item[0] for item in image_path_labels]
    image_labels = [int(item[1]) for item in image_path_labels]
    return image_path_list, image_labels

def save_image_path_label_to_txt(image_path_list, image_labels, patch_list_txt):
    image_path_label_list = ['%s %d' % (item, idx) for idx, item in
                             zip(image_labels, image_path_list)]
    io_function.save_list_to_txt(patch_list_txt, image_path_label_list)


def read_sub_image_labels_one_region(save_img_dir, para_file, area_ini, b_training=True):
    '''
    read image patches and labels (if available) from one region for image classification (no need to extract from big imagery using vector file)
    :param save_img_dir: save directory (only save a txt file, containing image path and label )
    :param para_file: para file
    :param area_ini: area file
    :param b_training: if True, will read image for training, otherwise, will read image for inference
    :return:
    '''

    area_name_remark_time = parameters.get_area_name_remark_time(area_ini)
    patch_list_txt = os.path.join(save_img_dir, area_name_remark_time + '_patch_list.txt')
    if os.path.isfile(patch_list_txt):
        print('%s exists, read it directly'%patch_list_txt)
        image_path_list, image_labels = read_image_path_label_from_txt(patch_list_txt)
        return image_path_list, image_labels, patch_list_txt

    # class ids for this specific regions
    class_labels = parameters.get_file_path_parameters(area_ini, 'class_labels')
    class_labels_main = parameters.get_file_path_parameters(para_file, 'class_labels')

    if b_training:
        # for training
        image_dir = parameters.get_directory(area_ini, 'input_image_dir')                             # train_image_dir
        # image_or_pattern = parameters.get_string_parameters(area_ini, 'input_image_or_pattern')       # train_image_or_pattern
        image_patch_labels = parameters.get_file_path_parameters(area_ini, 'input_image_patch_labels')
    else:
        # for inference
        image_dir = parameters.get_directory(area_ini, 'inf_image_dir')                             # inf_image_dir
        # image_or_pattern = parameters.get_string_parameters(area_ini, 'inf_image_or_pattern')       # inf_image_or_pattern
        image_patch_labels = parameters.get_file_path_parameters(area_ini, 'inf_image_patch_labels')

    image_path_labels = [item.split() for item in io_function.read_list_from_txt(image_patch_labels)]
    # image_path_labels = image_path_labels[:200] # for test
    image_path_list = [os.path.join(image_dir, item[0]) for item in image_path_labels]

    # check path
    if os.path.isfile(image_path_list[0]) is False:
        raise IOError("%s doesn't exists, please check the image directory in %s"%(image_path_list[0],area_ini))

    image_labels = [int(item[1]) for item in image_path_labels]

    class_id_original = read_label_ids_local(class_labels)

    if class_labels_main != class_labels:
        image_labels = convert_label_id_to_newSystem(image_labels, class_id_original)

    if os.path.isfile(patch_list_txt) is False:
        # save the relative path and label to file
        save_image_path_label_to_txt(image_labels, image_path_list, patch_list_txt)


    return image_path_list, image_labels, patch_list_txt

def extract_sub_image_labels_one_region(save_img_dir, para_file, area_ini, b_training=True, b_convert_label=True):
    '''
     get some images and labels (if available) from one region for image classification
    :param save_img_dir:  save directory for the saved images
    :param para_file: para_file
    :param area_ini: area file
    :param b_training: if True, will extract image for training, otherwise, will extract image for inference
    :return: image_path_list, image_labels, patch_list_txt (file_path label_value)
    '''

    # extract sub-images
    get_subImage_script = os.path.join(code_dir, 'datasets', 'get_subImages.py')
    extract_img_dir = save_img_dir

    dstnodata = parameters.get_string_parameters(para_file, 'dst_nodata')
    buffersize = parameters.get_string_parameters(para_file, 'buffer_size')
    rectangle_ext = parameters.get_string_parameters(para_file, 'b_use_rectangle')
    process_num = parameters.get_digit_parameters(para_file, 'process_num', 'int')

    area_name_remark_time = parameters.get_area_name_remark_time(area_ini)
    patch_list_txt = os.path.join(extract_img_dir, area_name_remark_time + '_patch_list.txt')

    if os.path.isfile(patch_list_txt):
        print('%s exists, read it directly' % patch_list_txt)
        image_path_labels = [item.split() for item in io_function.read_list_from_txt(patch_list_txt)]
        image_path_list = [item[0] for item in image_path_labels]
        image_labels = [int(item[1]) for item in image_path_labels]
        return image_path_list, image_labels, patch_list_txt

    if b_training:
        # for training
        image_dir = parameters.get_directory(area_ini, 'input_image_dir')                             # train_image_dir
        image_or_pattern = parameters.get_string_parameters(area_ini, 'input_image_or_pattern')       # train_image_or_pattern
        all_polygons_labels = parameters.get_file_path_parameters_None_if_absence(area_ini, 'training_polygons')    # training_polygons
    else:
        # for inference
        image_dir = parameters.get_directory(area_ini, 'inf_image_dir')                             # inf_image_dir
        image_or_pattern = parameters.get_string_parameters(area_ini, 'inf_image_or_pattern')       # inf_image_or_pattern
        all_polygons_labels = parameters.get_file_path_parameters_None_if_absence(area_ini, 'all_polygons_labels')


    extract_done_indicator = os.path.join(extract_img_dir, 'extract_image_using_vector.done')


    if all_polygons_labels is not None:
        command_string = get_subImage_script + ' -b ' + str(buffersize) + ' -e ' + image_or_pattern + \
                         ' -o ' + extract_img_dir + ' -n ' + str(dstnodata) + ' -p ' + str(process_num) \
                         + ' ' + rectangle_ext + ' --no_label_image ' + all_polygons_labels + ' ' + image_dir
        if os.path.isfile(extract_done_indicator):
            basic.outputlogMessage('Warning, sub-images already been extracted, read them directly')
        else:
            basic.os_system_exit_code(command_string)
        image_path_list = io_function.get_file_list_by_pattern(extract_img_dir, 'subImages/*.tif')
        image_labels = class_utils.get_class_labels_from_vector_file(image_path_list, all_polygons_labels)
    else:
        # get sub-images, grid by grid
        all_polygons_dir = parameters.get_directory(area_ini, 'all_polygons_dir')
        all_polygons_pattern = parameters.get_string_parameters(area_ini, 'all_polygons_pattern')
        vector_file_list = class_utils.get_file_list(all_polygons_dir, all_polygons_pattern, area_ini)
        raster_file_list = class_utils.get_file_list(image_dir, image_or_pattern, area_ini)

        image_path_list = []
        image_labels = []

        # pair the vector file and raster files
        raster_vector_pairs = class_utils.pair_raster_vecor_files_grid(vector_file_list, raster_file_list)
        for key in raster_vector_pairs:
            vector_file = raster_vector_pairs[key][0]
            raster_file = raster_vector_pairs[key][1]
            grid_save_dir = os.path.join(extract_img_dir, 'grid%d' % key)
            command_string = get_subImage_script + ' -b ' + str(buffersize) + ' -e ' + os.path.basename(raster_file) + \
                             ' -o ' + grid_save_dir + ' -n ' + str(dstnodata) + ' -p ' + str(process_num) \
                             + ' ' + rectangle_ext + ' --no_label_image ' + vector_file + ' ' + os.path.dirname(raster_file)
            if os.path.isfile(extract_done_indicator):
                basic.outputlogMessage('Warning, sub-images already been extracted, read them directly')
            else:
                basic.os_system_exit_code(command_string)

            image_path_list_grid = io_function.get_file_list_by_pattern(grid_save_dir, 'subImages/*.tif')
            image_labels_grid = class_utils.get_class_labels_from_vector_file(image_path_list_grid, vector_file)

            image_path_list.extend(image_path_list_grid)
            image_labels.extend(image_labels_grid)

    if b_convert_label:
        image_labels = convert_label_id_to_newSystem(image_labels, class_id_shp)

    if os.path.isfile(patch_list_txt) is False:
        # save the relative path and label to file
        image_path_label_list = ['%s %d' % (os.path.relpath(item), idx) for idx, item in
                                 zip(image_labels, image_path_list)]
        io_function.save_list_to_txt(patch_list_txt, image_path_label_list)

    if os.path.isfile(extract_done_indicator) is False:
        with open(extract_done_indicator, 'w') as f_obj:
            f_obj.writelines('%s image extracting, complete on %s \n' % (extract_img_dir, timeTools.get_now_time_str()))

    return image_path_list, image_labels, patch_list_txt


def merge_imagePatch_labels_for_multi_regions(image_patch_labels_list_txts, save_path):
    with open(save_path, 'w') as outfile:
        for fname in image_patch_labels_list_txts:
            with open(fname) as infile:
                outfile.write(infile.read())

def get_sub_images_multi_regions_for_training(WORK_DIR, para_file):
    print("get and organize training data for image classification")

    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    SECONDS = time.time()

    training_regions = parameters.get_string_list_parameters_None_if_absence(para_file, 'training_regions')
    if training_regions is None or len(training_regions) < 1:
        raise ValueError('No training area is set in %s' % para_file)

    # read class name and ids
    class_labels = parameters.get_file_path_parameters(para_file,'class_labels')
    read_label_ids(class_labels)

    image_patch_labels_list_txts = []
    training_data_dir = class_utils.get_training_data_dir(WORK_DIR)

    for area_idx, area_ini in enumerate(training_regions):
        basic.outputlogMessage(' %d/%d: getting training data from region: %s'%(area_idx, len(training_regions), area_ini))
        area_name_remark_time = parameters.get_area_name_remark_time(area_ini)

        extract_img_dir = os.path.join(training_data_dir, area_name_remark_time)
        if os.path.isdir(extract_img_dir) is False:
            io_function.mkdir(extract_img_dir)
        area_data_type = parameters.get_string_parameters(area_ini, 'area_data_type')
        if area_data_type == 'image_patch':
            # directly read

            image_path_list, image_labels, patch_list_txt = read_sub_image_labels_one_region(extract_img_dir, para_file, area_ini, b_training=True)
            image_patch_labels_list_txts.append(patch_list_txt)

        else:

            image_path_list, image_labels, patch_list_txt = \
                extract_sub_image_labels_one_region(extract_img_dir, para_file, area_ini, b_training=True, b_convert_label=True)
            image_patch_labels_list_txts.append(patch_list_txt)

    expr_name = parameters.get_string_parameters(para_file, 'expr_name')

    # merge label list
    save_path = class_utils.get_merged_training_data_txt(training_data_dir, expr_name, len(training_regions))
    merge_imagePatch_labels_for_multi_regions(image_patch_labels_list_txts, save_path)

    a_few_shot_samp_count = parameters.get_digit_parameters_None_if_absence(para_file,'a_few_shot_samp_count','int')
    if a_few_shot_samp_count is not None:
        # backup the original file
        save_path_all_samp = io_function.get_name_by_adding_tail(save_path,'all')
        io_function.copy_file_to_dst(save_path,save_path_all_samp)
        randomly_select_k_samples_each_classes(save_path_all_samp, save_path,sample_count=a_few_shot_samp_count)


    duration= time.time() - SECONDS
    os.system('echo "$(date): time cost of getting training data (image classification): %.2f seconds">>time_cost.txt'%duration)




def main(options, args):

    para_file = args[0]
    WORK_DIR = os.getcwd()
    get_sub_images_multi_regions_for_training(WORK_DIR,para_file)

if __name__ == '__main__':

    usage = "usage: %prog [options] para_file "
    parser = OptionParser(usage=usage, version="1.0 2024-04-26")
    parser.description = 'Introduction: extract sub-images and sub-labels '

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)