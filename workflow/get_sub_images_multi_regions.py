#!/usr/bin/env python
# Filename: get_sub_images_multi_files 
"""
introduction:  extract sub-images and sub-labels for one or multi given shape file (training polygons)

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 25 December, 2019
modified on 19 January, 2021
"""

import io
import os,sys
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters

import utility.get_valid_percent_entropy as get_valid_percent_entropy

import basic_src.io_function as io_function
import datasets.raster_io as raster_io
import time

import datasets.get_subImages_json as get_subImages_json
import datasets.rasterize_polygons as rasterize_polygons

def get_subImage_subLabel_one_shp(get_subImage_script,all_train_shp, buffersize, dstnodata, rectangle_ext, train_shp,
                                  input_image_dir, file_pattern = None, process_num=1):
    if file_pattern is None:
        file_pattern = '*.tif'

    command_string = get_subImage_script + ' -f ' + all_train_shp + ' -b ' + str(buffersize) + ' -e ' + file_pattern + \
                    ' -o ' + os.getcwd() + ' -n ' + str(dstnodata)  + ' -p ' + str(process_num) \
                     + ' ' + rectangle_ext + ' ' + train_shp + ' '+ input_image_dir

    # ${eo_dir}/sentinelScripts/get_subImages.py -f ${all_train_shp} -b ${buffersize} -e .tif \
    #             -o ${PWD} -n ${dstnodata} -r ${train_shp} ${input_image_dir}

    # status, result = basic.exec_command_string(command_string)  # this will wait command finished
    # os.system(command_string + "&")  # don't know when it finished
    res = os.system(command_string )      # this work
    if res != 0:
        sys.exit(1)

def get_subImage_one_shp(get_subImage_script,all_train_shp, buffersize, dstnodata, rectangle_ext, train_shp,
                                  input_image_dir, file_pattern = None, process_num=1):
    if file_pattern is None:
        file_pattern = '*.tif'

    command_string = get_subImage_script + ' -f ' + all_train_shp + ' -b ' + str(buffersize) + ' -e ' + file_pattern + \
                    ' -o ' + os.getcwd() + ' -n ' + str(dstnodata)  + ' -p ' + str(process_num) \
                     + ' ' + rectangle_ext +' --no_label_image '  + train_shp + ' '+ input_image_dir

    # ${eo_dir}/sentinelScripts/get_subImages.py -f ${all_train_shp} -b ${buffersize} -e .tif \
    #             -o ${PWD} -n ${dstnodata} -r ${train_shp} ${input_image_dir}

    # status, result = basic.exec_command_string(command_string)  # this will wait command finished
    # os.system(command_string + "&")  # don't know when it finished
    res = os.system(command_string )      # this work
    if res != 0:
        sys.exit(1)

def copy_subImages_labels_directly(subImage_dir,subLabel_dir,area_ini):

    input_image_dir = parameters.get_directory_None_if_absence(area_ini, 'input_image_dir')
    # it is ok consider a file name as pattern and pass it the following functions to get file list
    input_image_or_pattern = parameters.get_string_parameters(area_ini, 'input_image_or_pattern')

    # label raster folder
    label_raster_dir = parameters.get_directory_None_if_absence(area_ini, 'label_raster_dir')
    sub_images_list = []
    label_path_list = []

    if os.path.isdir(subImage_dir) is False:
        io_function.mkdir(subImage_dir)
    if os.path.isdir(subLabel_dir) is False:
        io_function.mkdir(subLabel_dir)

    sub_images = io_function.get_file_list_by_pattern(input_image_dir,input_image_or_pattern)
    for sub_img in sub_images:
        # find the corresponding label raster
        label_name = io_function.get_name_by_adding_tail(os.path.basename(sub_img),'label')
        label_path = os.path.join(label_raster_dir,label_name)
        if os.path.isfile(label_path):
            sub_images_list.append(sub_img)
            label_path_list.append(label_path)
        else:
            print('Warning, cannot find label for %s in %s'%(sub_img,label_raster_dir))


    # copy sub-images, adding to txt files
    with open('sub_images_labels_list.txt','a') as f_obj:
        for tif_path, label_file in zip(sub_images_list, label_path_list):
            if label_file is None:
                continue
            dst_subImg = os.path.join(subImage_dir, os.path.basename(tif_path))

            # copy sub-images
            io_function.copy_file_to_dst(tif_path,dst_subImg, overwrite=True)

            dst_label_file = os.path.join(subLabel_dir, os.path.basename(label_file))
            io_function.copy_file_to_dst(label_file, dst_label_file, overwrite=True)

            sub_image_label_str = dst_subImg + ":" + dst_label_file + '\n'
            f_obj.writelines(sub_image_label_str)


def get_sub_images_multi_regions(para_file):

    print("extract sub-images and sub-labels for a given shape file (training polygons)")

    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    get_subImage_script = os.path.join(code_dir, 'datasets', 'get_subImages.py')
    SECONDS = time.time()

    # get name of training areas
    multi_training_regions = parameters.get_string_list_parameters_None_if_absence(para_file, 'training_regions')
    if multi_training_regions is None or len(multi_training_regions) < 1:
        raise ValueError('No training area is set in %s'%para_file)

    # multi_training_files = parameters.get_string_parameters_None_if_absence(para_file, 'multi_training_files')

    dstnodata = parameters.get_string_parameters(para_file, 'dst_nodata')
    buffersize = parameters.get_string_parameters(para_file, 'buffer_size')
    rectangle_ext = parameters.get_string_parameters(para_file, 'b_use_rectangle')
    process_num = parameters.get_digit_parameters(para_file,'process_num', 'int')

    b_no_label_image = parameters.get_bool_parameters_None_if_absence(para_file,'b_no_label_image')

    if os.path.isfile('sub_images_labels_list.txt'):
        io_function.delete_file_or_dir('sub_images_labels_list.txt')

    subImage_dir = parameters.get_string_parameters_None_if_absence(para_file,'input_train_dir')
    subLabel_dir = parameters.get_string_parameters_None_if_absence(para_file,'input_label_dir')

    # record sub_images_labels from each area_ini
    area_ini_sub_images_labels = {}
    sub_image_label_list_before = []    # the list before getting sub-images

    # loop each training regions
    for idx, area_ini in enumerate(multi_training_regions):

        input_image_dir = parameters.get_directory_None_if_absence(area_ini, 'input_image_dir')

        # it is ok consider a file name as pattern and pass it the following functions to get file list
        input_image_or_pattern = parameters.get_string_parameters(area_ini, 'input_image_or_pattern')

        b_sub_images_json = parameters.get_bool_parameters_None_if_absence(area_ini,'b_sub_images_json')
        b_label_raster_aval = parameters.get_bool_parameters_None_if_absence(area_ini,'b_label_raster_aval')
        b_polygons_for_entire_scene = parameters.get_bool_parameters_None_if_absence(area_ini,'b_polygons_for_entire_scene')
        if b_sub_images_json is True:
            # copy sub-images, then covert json files to label images.
            object_names = parameters.get_string_list_parameters(para_file,'object_names')
            get_subImages_json.get_subimages_label_josn(input_image_dir,input_image_or_pattern,subImage_dir,subLabel_dir,object_names,
                                                        b_no_label_image=b_no_label_image,process_num=process_num)

            pass
        elif b_label_raster_aval is True:
            # copy the label raster and images directly.
            copy_subImages_labels_directly(subImage_dir,subLabel_dir,area_ini)
        elif b_polygons_for_entire_scene is True:
            # get label raster for entire scenes (not extract sub-images) by using rasterizing
            input_polygon_dir = parameters.get_string_parameters(area_ini, 'input_polygon_dir')
            input_polygon_or_pattern = parameters.get_string_parameters(area_ini,'input_polygon_or_pattern')
            rasterize_polygons.get_subimages_SpaceNet(input_image_dir,input_image_or_pattern,input_polygon_dir,input_polygon_or_pattern,subImage_dir,
                                                      subLabel_dir,burn_value=1)
            pass

        else:

            all_train_shp = parameters.get_file_path_parameters_None_if_absence(area_ini, 'training_polygons')
            train_shp = parameters.get_string_parameters(area_ini, 'training_polygons_sub')

            # get subImage and subLabel for one training polygons
            print('extract training data from image folder (%s) and polgyons (%s)' % (input_image_dir, train_shp))
            if b_no_label_image is True:
                get_subImage_one_shp(get_subImage_script,all_train_shp, buffersize, dstnodata, rectangle_ext, train_shp,
                                          input_image_dir, file_pattern=input_image_or_pattern, process_num=process_num)
            else:
                get_subImage_subLabel_one_shp(get_subImage_script,all_train_shp, buffersize, dstnodata, rectangle_ext, train_shp,
                                          input_image_dir, file_pattern=input_image_or_pattern, process_num=process_num)

        sub_image_label_list_after = io_function.read_list_from_txt('sub_images_labels_list.txt')
        area_ini_sub_images_labels[area_ini] = sub_image_label_list_after[len(sub_image_label_list_before):]
        # update list
        sub_image_label_list_before = sub_image_label_list_after

    # as the file names in sub_images_labels_list.txt was changed to base name, so add subImage_dir and subLabel_dir back
    sub_image_label_str_list_new = []
    with open('sub_images_labels_list.txt', 'r') as f_obj:
        lines = f_obj.readlines()
        for line in lines:
            image_path, label_path = line.strip().split(':')
            if os.path.isfile(image_path):
                break
            else:
                image_path_new = os.path.join(subImage_dir, image_path)
                label_path_new = os.path.join(subLabel_dir, label_path)
                sub_image_label_str = image_path_new + ":" + label_path_new # + '\n' use "save_list_to_txt", don't need '\n' here
                sub_image_label_str_list_new.append(sub_image_label_str)
    io_function.save_list_to_txt('sub_images_labels_list.txt',sub_image_label_str_list_new)

    # check black sub-images or most part of the sub-images is black (nodata)
    new_sub_image_label_list = []
    delete_sub_image_label_list = []
    subImage_dir_delete = subImage_dir + '_delete'
    subLabel_dir_delete = subLabel_dir + '_delete'
    io_function.mkdir(subImage_dir_delete)
    if b_no_label_image is None or b_no_label_image is False:
        io_function.mkdir(subLabel_dir_delete)
    b_check_sub_image_quality = parameters.get_bool_parameters_None_if_absence(para_file,'b_check_sub_image_quality')
    if b_check_sub_image_quality is True:
        get_valid_percent_entropy.plot_valid_entropy(subImage_dir)
        with open('sub_images_labels_list.txt','r') as f_obj:
            lines = f_obj.readlines()
            for line in lines:
                image_path, label_path = line.strip().split(':')
                # valid_per = raster_io.get_valid_pixel_percentage(image_path)
                valid_per, entropy = raster_io.get_valid_percent_shannon_entropy(image_path)    # base=10
                if valid_per > 60 and entropy >= 0.5:
                    new_sub_image_label_list.append(line)
                else:
                    delete_sub_image_label_list.append(line)
                    io_function.movefiletodir(image_path,subImage_dir_delete)
                    if os.path.isfile(label_path):
                        io_function.movefiletodir(label_path,subLabel_dir_delete)
    else:
        with open('sub_images_labels_list.txt','r') as f_obj:
            new_sub_image_label_list = f_obj.readlines()

    if len(delete_sub_image_label_list) > 0:
        with open('sub_images_labels_list.txt', 'w') as f_obj:
            for line in new_sub_image_label_list:
                f_obj.writelines(line)

    for del_line in delete_sub_image_label_list:
        for idx, area_ini in enumerate(multi_training_regions):
            if del_line in area_ini_sub_images_labels[area_ini]:
                area_ini_sub_images_labels[area_ini].remove(del_line)

    io_function.save_dict_to_txt_json('area_ini_sub_images_labels.txt', area_ini_sub_images_labels)

    # check weather they have the same subImage and subLabel
    if b_no_label_image is None or b_no_label_image is False:
        sub_image_list = io_function.get_file_list_by_pattern(subImage_dir,'*.tif')
        sub_label_list = io_function.get_file_list_by_pattern(subLabel_dir,'*.tif')
        if len(sub_image_list) != len(sub_label_list):
            raise ValueError('the count of subImage (%d) and subLabel (%d) is different'
                             %(len(sub_image_list),len(sub_label_list)))

    # save brief information of sub-images
    height_list = []
    width_list = []
    band_count = 0
    dtype = 'unknown'
    for line in new_sub_image_label_list:
        image_path, label_path = line.strip().split(':')
        height, width, band_count, dtype = raster_io.get_height_width_bandnum_dtype(image_path)
        height_list.append(height)
        width_list.append(width)
    if len(height_list) < 1 or len(width_list) < 1:
        raise ValueError('No sub-images')
    # save info to file, if it exists, it will be overwritten
    img_count = len(new_sub_image_label_list)
    with open('sub_images_patches_info.txt','w') as f_obj:
        f_obj.writelines('information of sub-images: \n')
        f_obj.writelines('number of sub-images : %d \n' % img_count)
        f_obj.writelines('band count : %d \n'%band_count)
        f_obj.writelines('data type : %s \n'%dtype)
        f_obj.writelines('maximum width and height: %d, %d \n'% (max(width_list), max(height_list)) )
        f_obj.writelines('minimum width and height: %d, %d \n'% (min(width_list), min(height_list)) )
        f_obj.writelines('mean width and height: %.2f, %.2f \n\n'% (sum(width_list)/img_count, sum(height_list)/img_count))


    duration= time.time() - SECONDS
    os.system('echo "$(date): time cost of getting sub images and labels: %.2f seconds">>time_cost.txt'%duration)


def main(options, args):

    para_file = args[0]
    get_sub_images_multi_regions(para_file)

if __name__ == '__main__':

    usage = "usage: %prog [options] para_file "
    parser = OptionParser(usage=usage, version="1.0 2021-01-19")
    parser.description = 'Introduction: extract sub-images and sub-labels '

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)

