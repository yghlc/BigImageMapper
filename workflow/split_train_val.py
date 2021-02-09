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

def get_image_with_height_list(sample_txt, img_ext, info_type='training'):
    height_list = []
    width_list = []
    band_count = 0
    image_path_list = []
    dtype = 'unknown'
    with open(sample_txt, 'r') as f_obj:
        lines = f_obj.readlines()
        for line in lines:
            image_path = line.strip() + img_ext
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
        f_obj.writelines('mean width and height: %.2f, %.2f \n'% (sum(width_list)/img_count, sum(height_list)/img_count))

    return True


if __name__ == '__main__':

    print("%s : split data set into training and validation" % os.path.basename(sys.argv[0]))

    para_file = sys.argv[1]
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

    command_string = script + ' -p ' + str(training_data_per) + \
                     ' -t ' + train_sample_txt  + \
                     ' -v ' + val_sample_txt  + \
                     ' --shuffle ' + all_img_list
    res = os.system(command_string)
    if res!=0:
        sys.exit(res)

    # save brief information of image patches
    img_ext = parameters.get_string_parameters_None_if_absence(para_file, 'split_image_format')

    get_image_with_height_list(os.path.join(dir,train_sample_txt), img_ext, info_type='training')

    get_image_with_height_list(os.path.join(dir,val_sample_txt), img_ext, info_type='validation')
