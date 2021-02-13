#!/usr/bin/env python
# Filename: split_sub_images_test.py 
"""
# run "pytest split_sub_images_test.py " or "pytest " for test, add " -s for allowing print out"
# "pytest can automatically search *_test.py files "

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 13 February, 2021
"""
import os, sys

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')   # get path of deeplab_train_test.py
print(code_dir)
sys.path.insert(0, code_dir)


import basic_src.io_function as io_function

from multiprocessing import Pool

work_dir = os.path.expanduser('~/codes/PycharmProjects/Landuse_DL/working_dir')
os.chdir(work_dir)
para_file = 'main_para.ini'


import workflow.split_sub_images as split_sub_images

class TestdeeplabTrainclass():

    if os.path.isdir('split_images'):
        io_function.delete_file_or_dir('split_images')
    if os.path.isdir('split_labels'):
        io_function.delete_file_or_dir('split_labels')

    io_function.mkdir('split_images')
    io_function.mkdir('split_labels')

    def test_split_a_pair_sub_image_label(self):

        ### split the training image to many small patch (480*480)
        patch_w= 160 # parameters.get_string_parameters(para_file,'train_patch_width')
        patch_h= 160 #parameters.get_string_parameters(para_file,'train_patch_height')

        # notes
        # set overlay as 80, then width or height of patches range from 240 to 320.
        # so it will generate more patches than 160 ones

        # overlay_x= 80 # parameters.get_string_parameters(para_file,'train_pixel_overlay_x')
        # overlay_y= 80 #parameters.get_string_parameters(para_file,'train_pixel_overlay_y')

        overlay_x= 160 # parameters.get_string_parameters(para_file,'train_pixel_overlay_x')
        overlay_y= 160 #parameters.get_string_parameters(para_file,'train_pixel_overlay_y')

        split_image_format= '.png' # parameters.get_string_parameters(para_file,'split_image_format')

        trainImg_dir= 'subImages' #  parameters.get_string_parameters(para_file,'input_train_dir')
        labelImg_dir= 'subLabels' # parameters.get_string_parameters(para_file,'input_label_dir')

        if os.path.isdir(trainImg_dir) is False:
            raise IOError('%s not in the current folder, please get subImages first'%trainImg_dir)
        if os.path.isdir(labelImg_dir) is False:
            raise IOError('%s not in the current folder, please get subImages first'%labelImg_dir)

        # sub_img_label_txt = 'sub_images_labels_list_test.txt'
        sub_img_label_txt = 'sub_images_labels_list_1.txt'
        if os.path.isfile(sub_img_label_txt) is False:
            raise IOError('%s not in the current folder, please get subImages first' % sub_img_label_txt)

        with open(sub_img_label_txt) as txt_obj:
            line_list = [name.strip() for name in txt_obj.readlines()]
            for line in line_list:
                split_sub_images.split_a_pair_sub_image_label(line, patch_w, patch_h, overlay_x, overlay_y, split_image_format)




if __name__ == '__main__':

    pass