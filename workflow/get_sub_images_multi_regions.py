#!/usr/bin/env python
# Filename: get_sub_images_multi_files 
"""
introduction:  extract sub-images and sub-labels for one or multi given shape file (training polygons)

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 25 December, 2019
modified on 19 January, 2021
"""

import os,sys

def get_subImage_subLabel_one_shp(all_train_shp, buffersize, dstnodata, rectangle_ext, train_shp, input_image_dir, file_pattern = None):
    if file_pattern is None:
        file_pattern = '*.tif'

    command_string = get_subImage_script + ' -f ' + all_train_shp + ' -b ' + str(buffersize) + ' -e ' + file_pattern + \
                    ' -o ' + os.getcwd() + ' -n ' + str(dstnodata) + ' ' + rectangle_ext + ' ' + train_shp + ' '+ input_image_dir

    # ${eo_dir}/sentinelScripts/get_subImages.py -f ${all_train_shp} -b ${buffersize} -e .tif \
    #             -o ${PWD} -n ${dstnodata} -r ${train_shp} ${input_image_dir}

    # status, result = basic.exec_command_string(command_string)  # this will wait command finished
    # os.system(command_string + "&")  # don't know when it finished
    os.system(command_string )      # this work

if __name__ == '__main__':

    print("%s : extract sub-images and sub-labels for a given shape file (training polygons)" %
          os.path.basename(sys.argv[0]))

    para_file = sys.argv[1]
    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    code_dir = os.path.join(os.path.dirname(sys.argv[0]), '..')
    sys.path.insert(0, code_dir)
    import parameters

    deeplabRS = parameters.get_directory_None_if_absence(para_file, 'deeplabRS_dir')
    sys.path.insert(0, deeplabRS)
    import basic_src.io_function as io_function

    eo_dir = code_dir
    get_subImage_script = os.path.join(eo_dir, 'datasets', 'get_subImages.py')

    # get name of training areas
    multi_training_regions = parameters.get_string_list_parameters_None_if_absence(para_file, 'training_regions')
    if multi_training_regions is None or len(multi_training_regions) < 1:
        raise ValueError('No training area is set in %s'%para_file)

    # multi_training_files = parameters.get_string_parameters_None_if_absence(para_file, 'multi_training_files')

    dstnodata = parameters.get_string_parameters(para_file, 'dst_nodata')
    buffersize = parameters.get_string_parameters(para_file, 'buffer_size')
    rectangle_ext = parameters.get_string_parameters(para_file, 'b_use_rectangle')

    if os.path.isfile('sub_images_labels_list.txt'):
        io_function.delete_file_or_dir('sub_images_labels_list.txt')

    subImage_dir = parameters.get_string_parameters_None_if_absence(para_file,'input_train_dir')
    subLabel_dir = parameters.get_string_parameters_None_if_absence(para_file,'input_label_dir')

    # loop each training regions
    for idx, area_ini in enumerate(multi_training_regions):

        input_image_dir = parameters.get_directory_None_if_absence(area_ini, 'input_image_dir')

        # it is ok consider a file name as pattern and pass it the following functions to get file list
        input_image_or_pattern = parameters.get_string_parameters(area_ini, 'input_image_or_pattern')

        all_train_shp = parameters.get_file_path_parameters_None_if_absence(area_ini, 'training_polygons')
        train_shp = parameters.get_string_parameters(area_ini, 'training_polygons_sub')

        # get subImage and subLabel for one training polygons
        print('extract training data from image folder (%s) and polgyons (%s)' % (input_image_dir, train_shp))
        get_subImage_subLabel_one_shp(all_train_shp, buffersize, dstnodata, rectangle_ext, train_shp,
                                      input_image_dir, file_pattern=input_image_or_pattern)


    # check weather they have the same subImage and subLabel
    sub_image_list = io_function.get_file_list_by_pattern(subImage_dir,'*.tif')
    sub_label_list = io_function.get_file_list_by_pattern(subLabel_dir,'*.tif')
    if len(sub_image_list) != len(sub_label_list):
        raise ValueError('the count of subImage (%d) and subLabel (%d) is different'
                         %(len(sub_image_list),len(sub_label_list)))
