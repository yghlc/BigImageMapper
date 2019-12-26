#!/usr/bin/env python
# Filename: split_sub_images 
"""
introduction: split sub-images and sub-labels

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 26 December, 2019
"""

import os,sys

print("%s : split sub-images and sub-labels"% os.path.basename(sys.argv[0]))

para_file=sys.argv[1]
if os.path.isfile(para_file) is False:
    raise IOError('File %s not exists in current folder: %s'%(para_file, os.getcwd()))

deeplabRS=os.path.expanduser('~/codes/PycharmProjects/DeeplabforRS')
sys.path.insert(0, deeplabRS)

import parameters
import basic_src.io_function as io_function

eo_dir=os.path.expanduser("~/codes/PycharmProjects/Landuse_DL")
# split_image_script=os.path.join(eo_dir,'grss_data_fusion', 'split_image.py')
sys.path.insert(0, eo_dir)
import grss_data_fusion.split_image as split_image

if os.path.isdir('split_images'):
    io_function.delete_file_or_dir('split_images')
if os.path.isdir('split_labels'):
    io_function.delete_file_or_dir('split_labels')

io_function.mkdir('split_images')
io_function.mkdir('split_labels')

### split the training image to many small patch (480*480)
patch_w=parameters.get_string_parameters(para_file,'train_patch_width')
patch_h=parameters.get_string_parameters(para_file,'train_patch_height')
overlay=parameters.get_string_parameters(para_file,'train_pixel_overlay_x')
split_image_format=parameters.get_string_parameters(para_file,'split_image_format')

trainImg_dir=parameters.get_string_parameters(para_file,'input_train_dir')
labelImg_dir=parameters.get_string_parameters(para_file,'input_label_dir')

def split_to_patches(image_path, out_dir, patch_width, patch_height, overlay_x, overlay_y, out_format, file_pre_name=None):
    patch_width = int(patch_width)
    patch_height = int(patch_height)
    overlay_x = int(overlay_x)
    overlay_y = int(overlay_y)

    if out_format == '.png': out_format = 'PNG'
    if out_format == '.tif': out_format = 'GTIFF'

    split_image.split_image(image_path, out_dir, patch_width, patch_height, overlay_x, overlay_y, out_format,pre_name=file_pre_name)



with open('sub_images_labels_list.txt') as txt_obj:
    line_list = [name.strip() for name in txt_obj.readlines()]
    for line in line_list:
        sub_image, sub_label = line.split(':')

        # split sub image
        split_to_patches(sub_image, 'split_images', patch_w, patch_h, overlay, overlay, split_image_format)

        # split sub label (change the file name to be the same as sub_image name)
        pre_name = os.path.splitext(os.path.basename(sub_image))[0]
        split_to_patches(sub_label, 'split_labels', patch_w, patch_h, overlay, overlay, split_image_format, file_pre_name=pre_name)

    # output trainval.txt and val.txt file
    files_list = io_function.get_file_list_by_ext(split_image_format, 'split_images',bsub_folder=False)
    io_function.mkdir('list')
    trainval = os.path.join('list','trainval.txt')
    val = os.path.join('list','val.txt')
    with open(trainval,'w') as w_obj:
        for file_name in files_list:
            w_obj.writelines(os.path.splitext(os.path.basename(file_name))[0] + '\n')

    io_function.copy_file_to_dst(trainval,val,overwrite=True)



