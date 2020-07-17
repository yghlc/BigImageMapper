#!/usr/bin/env python
# Filename: get_sub_images_multi_files 
"""
introduction:  extract sub-images and sub-labels for one or multi given shape file (training polygons)

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 25 December, 2019
"""

import os,sys

print("%s : extract sub-images and sub-labels for a given shape file (training polygons)"% os.path.basename(sys.argv[0]))

para_file=sys.argv[1]
if os.path.isfile(para_file) is False:
    raise IOError('File %s not exists in current folder: %s'%(para_file, os.getcwd()))

deeplabRS=os.path.expanduser('~/codes/PycharmProjects/DeeplabforRS')
sys.path.insert(0, deeplabRS)

import parameters
import basic_src.io_function as io_function

eo_dir=os.path.expanduser("~/codes/PycharmProjects/Landuse_DL")
get_subImage_script=os.path.join(eo_dir,'sentinelScripts', 'get_subImages.py')

multi_training_files = parameters.get_string_parameters_None_if_absence(para_file,'multi_training_files')

input_image_dir = parameters.get_string_parameters(para_file,'input_image_dir')

dstnodata = parameters.get_string_parameters(para_file, 'dst_nodata')
buffersize = parameters.get_string_parameters(para_file, 'buffer_size')
rectangle_ext = parameters.get_string_parameters(para_file, 'b_use_rectangle')

if os.path.isfile('sub_images_labels_list.txt'):
    io_function.delete_file_or_dir('sub_images_labels_list.txt')

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

if multi_training_files is None:
    all_train_shp = parameters.get_string_parameters(para_file, 'training_polygons')
    train_shp = parameters.get_string_parameters(para_file, 'training_polygons_sub')

    # get subImage and subLabel for one training polygons
    print('extract training data from image folder (%s) and polgyons (%s)' % (input_image_dir, train_shp))

    get_subImage_subLabel_one_shp(all_train_shp, buffersize, dstnodata, rectangle_ext, train_shp, input_image_dir)

else:
    # get subImage and subLabel for multi training polygons
    with open(multi_training_files, 'r') as txt_obj:
        line_list = [name.strip() for name in txt_obj.readlines() if len(name) > 1]

    # if the full set of training polygons exists
    line_contain_all_train_shp = None
    training_files_allPolygons = io_function.get_name_by_adding_tail(multi_training_files, 'allPolygons')
    if os.path.isfile(training_files_allPolygons):
        with open(training_files_allPolygons, 'r') as txt_obj:
            line_contain_all_train_shp = [name.strip() for name in txt_obj.readlines() if len(name) > 1]
        if len(line_contain_all_train_shp) != len(line_list):
            raise ValueError('The count of all_train_shp is not equal to the one of train_shp')

    for idx in range(len(line_list)):
        line = line_list[idx]
        folder, pattern, train_polygon_shp = line.split(':')

        if line_contain_all_train_shp is not None:
            folder, pattern, all_train_shp = line_contain_all_train_shp[idx].split(':')
        else:
            all_train_shp = train_polygon_shp

        image_folder = os.path.join(input_image_dir,folder)
        print('extract training data from image folder (%s) and polgyons (%s)' % (image_folder, train_polygon_shp))

        get_subImage_subLabel_one_shp(all_train_shp, buffersize, dstnodata, rectangle_ext, train_polygon_shp, image_folder, file_pattern=pattern)


# check weather they have the same subImage and subLabel
sub_image_list = io_function.get_file_list_by_pattern('subImages','*.tif')
sub_label_list = io_function.get_file_list_by_pattern('subLabels','*.tif')
if len(sub_image_list) != len(sub_label_list):
    raise ValueError('the count of subImage (%d) and subLabel (%d) is different'
                     %(len(sub_image_list),len(sub_label_list)))
