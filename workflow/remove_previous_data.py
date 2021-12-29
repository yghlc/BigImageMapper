#!/usr/bin/env python
# Filename: remove_previous_data.py 
"""
introduction: remove previous data to run again.

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 18 January, 2021
"""

import os,sys

# print(os.path.abspath(sys.argv[0]))
code_dir = os.path.join(os.path.dirname(sys.argv[0]), '..')
sys.path.insert(0, code_dir)
import parameters

import basic_src.io_function as io_function

def remove_previous_data(para_file):

    print("remove previous data or results to run again" )

    if os.path.isfile(para_file) is False:
        raise IOError('File %s does not exists in current folder: %s' % (para_file, os.getcwd()))

    subImage_dir = parameters.get_string_parameters_None_if_absence(para_file,'input_train_dir')
    subLabel_dir = parameters.get_string_parameters_None_if_absence(para_file,'input_label_dir')

    if os.path.isdir(subImage_dir):
        io_function.delete_file_or_dir(subImage_dir)
        print('remove %s' % subImage_dir)
    if os.path.isdir(subLabel_dir):
        io_function.delete_file_or_dir(subLabel_dir)
        print('remove %s' % subLabel_dir)

    subImage_dir_delete = subImage_dir + '_delete'
    subLabel_dir_delete = subLabel_dir + '_delete'
    if os.path.isdir(subImage_dir_delete):
        io_function.delete_file_or_dir(subImage_dir_delete)
        print('remove %s' % subImage_dir_delete)
    if os.path.isdir(subLabel_dir_delete):
        io_function.delete_file_or_dir(subLabel_dir_delete)
        print('remove %s '% subLabel_dir_delete)

    if os.path.isdir('train_split_images'):
        io_function.delete_file_or_dir('train_split_images')
        print('remove %s '% 'train_split_images')
    if os.path.isdir('train_split_labels'):
        io_function.delete_file_or_dir('train_split_labels')
        print('remove %s ' % 'train_split_labels')
        
    if os.path.isdir('val_split_images'):
        io_function.delete_file_or_dir('val_split_images')
        print('remove %s '% 'val_split_images')
    if os.path.isdir('val_split_labels'):
        io_function.delete_file_or_dir('val_split_labels')
        print('remove %s ' % 'val_split_labels')
        
    if os.path.isdir('list'):
        io_function.delete_file_or_dir('list')
        print('remove %s '% 'list')

    images_including_aug= os.path.join('list', 'images_including_aug.txt')
    if os.path.isfile(images_including_aug):
        io_function.delete_file_or_dir(images_including_aug)
        print('remove %s ' % 'list/images_including_aug.txt')

    if os.path.isdir('tfrecord'):
        io_function.delete_file_or_dir('tfrecord')
        print('remove %s ' % 'tfrecord')

if __name__ == '__main__':
    para_file = sys.argv[1]
    remove_previous_data(para_file)








