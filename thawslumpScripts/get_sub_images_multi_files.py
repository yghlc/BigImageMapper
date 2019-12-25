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

eo_dir=os.path.expanduser("~/codes/PycharmProjects/Landuse_DL")
get_subImage_script=os.path.join(eo_dir,'sentinelScripts', 'get_subImages.py')

multi_training_files = parameters.get_string_parameters_None_if_absence(para_file,'multi_training_files')

input_image_dir = parameters.get_string_parameters(para_file,'input_image_dir')

dstnodata = parameters.get_string_parameters(para_file, 'dst_nodata')
buffersize = parameters.get_string_parameters(para_file, 'buffer_size')
rectangle_ext = parameters.get_string_parameters(para_file, 'b_use_rectangle')


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
        line_list = [name.strip() for name in txt_obj.readlines()]
    for line in line_list:
        folder, pattern, train_polygon_shp = line.split(':')
        image_folder = os.path.join(input_image_dir,folder)
        print('extract training data from image folder (%s) and polgyons (%s)' % (image_folder, train_polygon_shp))

        get_subImage_subLabel_one_shp(train_polygon_shp, buffersize, dstnodata, rectangle_ext, train_polygon_shp, image_folder, file_pattern=pattern)


