#!/usr/bin/env python
# Filename: split_sub_images 
"""
introduction: split sub-images and sub-labels

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 26 December, 2019
modified: 19 January, 2021
"""

import os,sys
from optparse import OptionParser
import time

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters
import datasets.split_image as split_image

import basic_src.io_function as io_function

import workflow.split_train_val as split_train_val

import multiprocessing
from multiprocessing import Pool

def split_to_patches(image_path, out_dir, patch_width, patch_height, overlay_x, overlay_y, out_format, file_pre_name=None):
    patch_width = int(patch_width)
    patch_height = int(patch_height)
    overlay_x = int(overlay_x)
    overlay_y = int(overlay_y)

    if out_format == '.png': out_format = 'PNG'
    if out_format == '.tif': out_format = 'GTIFF'

    split_image.split_image(image_path, out_dir, patch_width, patch_height, overlay_x, overlay_y, out_format, pre_name=file_pre_name)

def split_a_pair_sub_image_label(image_txt, label_txt, line, patch_w, patch_h, overlay_x, overlay_y, split_image_format):

    sub_image, sub_label = line.split(':')
    # split sub image
    split_to_patches(sub_image, image_txt, patch_w, patch_h, overlay_x, overlay_y, split_image_format)

    # split sub label (change the file name to be the same as sub_image name)
    pre_name = os.path.splitext(os.path.basename(sub_image))[0]
    split_to_patches(sub_label, label_txt, patch_w, patch_h, overlay_x, overlay_y, split_image_format, file_pre_name=pre_name)

def split_sub_images(para_file):
    print("split sub-images and sub-labels")

    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s'%(para_file, os.getcwd()))

    SECONDS = time.time()
    
    if os.path.isdir('train_split_images'):
        io_function.delete_file_or_dir('train_split_images')
    if os.path.isdir('train_split_labels'):
        io_function.delete_file_or_dir('train_split_labels')

    io_function.mkdir('train_split_images')
    io_function.mkdir('train_split_labels')
    
    
    if os.path.isdir('val_split_images'):
        io_function.delete_file_or_dir('val_split_images')
    if os.path.isdir('val_split_labels'):
        io_function.delete_file_or_dir('val_split_labels')

    io_function.mkdir('val_split_images')
    io_function.mkdir('val_split_labels')
    
    train_split_image_txt = 'train_split_images'
    train_split_label_txt = 'train_split_labels'
    val_split_image_txt = 'val_split_images'
    val_split_label_txt = 'val_split_labels'

    ### split the training image to many small patch (480*480)
    patch_w=parameters.get_string_parameters(para_file,'train_patch_width')
    patch_h=parameters.get_string_parameters(para_file,'train_patch_height')
    overlay_x=parameters.get_string_parameters(para_file,'train_pixel_overlay_x')
    overlay_y=parameters.get_string_parameters(para_file,'train_pixel_overlay_y')
    split_image_format=parameters.get_string_parameters(para_file,'split_image_format')

    trainImg_dir=parameters.get_string_parameters(para_file,'input_train_dir')
    labelImg_dir=parameters.get_string_parameters(para_file,'input_label_dir')
    proc_num = parameters.get_digit_parameters(para_file,'process_num','int')
    
    train_sample_txt = parameters.get_string_parameters(para_file, 'training_sample_list_txt')
    val_sample_txt = parameters.get_string_parameters(para_file, 'validation_sample_list_txt')

    if os.path.isdir(trainImg_dir) is False:
        raise IOError('%s not in the current folder, please get subImages first'%trainImg_dir)
    if os.path.isdir(labelImg_dir) is False:
        raise IOError('%s not in the current folder, please get subImages first'%labelImg_dir)
        
        
    train_sub_img_label_txt = os.path.join('list','training_sub_images_labels.txt')
    val_sub_img_label_txt = os.path.join('list','validation_sub_images_labels.txt')
    
    if os.path.isfile(train_sub_img_label_txt) is False:
        raise IOError('%s not in the current folder, please get subImages first' % train_sub_img_label_txt)
    if os.path.isfile(val_sub_img_label_txt) is False:
        raise IOError('%s not in the current folder, please get subImages first' % val_sub_img_label_txt)

    with open(train_sub_img_label_txt) as txt_obj:
        line_list = [name.strip() for name in txt_obj.readlines()]
        
        parameters_list = [(train_split_image_txt, train_split_label_txt, line, patch_w, patch_h, overlay_x, overlay_y, split_image_format) for line in line_list]
        theadPool = Pool(proc_num)  # multi processes
        results = theadPool.starmap(split_a_pair_sub_image_label, parameters_list)  # need python3

        # output trainval.txt and val.txt file
        files_list = io_function.get_file_list_by_ext(split_image_format, train_split_image_txt,bsub_folder=False)
        train = os.path.join('list',train_sample_txt)
        with open(train,'w') as w_obj:
            for file_name in files_list:
                w_obj.writelines(os.path.splitext(os.path.basename(file_name))[0] + '\n')

    with open(val_sub_img_label_txt) as txt_obj:
        line_list = [name.strip() for name in txt_obj.readlines()]
        
        parameters_list = [(val_split_image_txt, val_split_label_txt, line, patch_w, patch_h, overlay_x, overlay_y, split_image_format) for line in line_list]
        theadPool = Pool(proc_num)  # multi processes
        results = theadPool.starmap(split_a_pair_sub_image_label, parameters_list)  # need python3

        # output trainval.txt and val.txt file
        files_list = io_function.get_file_list_by_ext(split_image_format, val_split_image_txt,bsub_folder=False)
        val = os.path.join('list',val_sample_txt)
        with open(val,'w') as w_obj:
            for file_name in files_list:
                w_obj.writelines(os.path.splitext(os.path.basename(file_name))[0] + '\n')

    duration= time.time() - SECONDS
    os.system('echo "$(date): time cost of splitting sub images and labels: %.2f seconds">>time_cost.txt'%duration)


def main(options, args):

    para_file=args[0]
    split_sub_images(para_file)

if __name__ == '__main__':

    usage = "usage: %prog [options] para_file"
    parser = OptionParser(usage=usage, version="1.0 2021-01-21")
    parser.description = 'Introduction: split sub-images '

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)





