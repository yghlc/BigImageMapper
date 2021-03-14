#!/usr/bin/env python
# Filename: split_train_val.py 
"""
introduction: split dataset to training and validation

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 19 January, 2021
"""

import os, sys

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)

import datasets.raster_io as raster_io
import re

def get_image_with_height_list(sample_txt, img_ext, info_type='training'):
    height_list = []
    width_list = []
    band_count = 0
    image_path_list = []
    dtype = 'unknown'
    with open(sample_txt, 'r') as f_obj:
        lines = f_obj.readlines()
        for line in lines:
            image_path = os.path.join('split_images', line.strip() + img_ext)
            height, width, band_count, dtype = raster_io.get_height_width_bandnum_dtype(image_path)
            image_path_list.append(image_path)
            height_list.append(height)
            width_list.append(width)

    # save info to file, if it exists, add information to the file
    img_count = len(image_path_list)
    with open('sub_images_patches_info.txt','a') as f_obj:
        f_obj.writelines('information of %s image patches: \n'%info_type)
        f_obj.writelines('number of %s image patches : %d \n' % (info_type,img_count))
        f_obj.writelines('band count : %d \n'%band_count)
        f_obj.writelines('data type : %s \n'%dtype)
        f_obj.writelines('maximum width and height: %d, %d \n'% (max(width_list), max(height_list)) )
        f_obj.writelines('minimum width and height: %d, %d \n'% (min(width_list), min(height_list)) )
        f_obj.writelines('mean width and height: %.2f, %.2f \n\n'% (sum(width_list)/img_count, sum(height_list)/img_count))

    return True

def get_sample_cout_of_each_class(sample_txt, info_type='training'):

    sample_count = {}
    with open(sample_txt, 'r') as f_obj:
        lines = f_obj.readlines()
        for line in lines:
            c_labels = re.findall(r'class_\d+',line)
            if len(c_labels) != 1:
                raise ValueError('class label (e.g,. class_1) is not correctly set in %s of file %s'%(line,sample_txt))
            c_label = c_labels[0]
            if c_label in sample_count.keys():
                sample_count[c_label] += 1
            else:
                sample_count[c_label] = 1

    # save info to file, if it exists, add information to the file
    with open('sub_images_patches_info.txt','a') as f_obj:
        f_obj.writelines('Sample count of each class in %s set: \n'%info_type)
        for key in sorted(sample_count.keys()):
            f_obj.writelines('Sample count of %s : %d \n' % (key, sample_count[key]))
        f_obj.writelines('\n')


    return True

def split_train_val(para_file):
    print("split data set into training and validation")

    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    code_dir = os.path.join(os.path.dirname(sys.argv[0]), '..')
    sys.path.insert(0, code_dir)
    import parameters

    script = os.path.join(code_dir, 'datasets', 'train_test_split.py')

    training_data_per = parameters.get_digit_parameters_None_if_absence(para_file, 'training_data_per','float')
    train_sample_txt = parameters.get_string_parameters(para_file, 'training_sample_list_txt')
    val_sample_txt = parameters.get_string_parameters(para_file, 'validation_sample_list_txt')

    dir = 'list'
    all_img_list = os.path.join(dir,'trainval.txt')

    # command_string = script + ' -p ' + str(training_data_per) + \
    #                  ' -t ' + train_sample_txt  + \
    #                  ' -v ' + val_sample_txt  + \
    #                  ' --shuffle ' + all_img_list
    # res = os.system(command_string)
    # if res!=0:
    #     sys.exit(1)

    Do_shuffle = True
    from datasets.train_test_split import train_test_split_main
    train_test_split_main(all_img_list,training_data_per,Do_shuffle,train_sample_txt,val_sample_txt)


    # save brief information of image patches
    img_ext = parameters.get_string_parameters_None_if_absence(para_file, 'split_image_format')

    get_image_with_height_list(os.path.join(dir,train_sample_txt), img_ext, info_type='training')

    get_image_with_height_list(os.path.join(dir,val_sample_txt), img_ext, info_type='validation')

    # save the count of each classes in training and validation
    get_sample_cout_of_each_class(os.path.join(dir,train_sample_txt), info_type='training')

    get_sample_cout_of_each_class(os.path.join(dir,val_sample_txt), info_type='validation')


if __name__ == '__main__':

    para_file = sys.argv[1]
    split_train_val(para_file)

