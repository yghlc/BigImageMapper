#!/usr/bin/env python
# Filename: train_test_split 
"""
introduction: Check the label images after data augmentation

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 27 November, 2018
"""

import sys,os
from optparse import OptionParser

import cv2
import numpy as np


HOME = os.path.expanduser('~')

# path of DeeplabforRS
codes_dir2 = HOME +'/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)
import parameters
import basic_src.basic as basic

basic.setlogfile('check_label_patches.txt')

def main(options, args):

    input_file = args[0]
    img_patch_dir = options.image_dir
    label_patch_dir = options.label_dir

    basic.outputlogMessage('check split image patches and label patches in %s, especially after data augmentation'%input_file)
    b_diff = False

    num_classes_noBG = parameters.get_digit_parameters(options.para_file,'NUM_CLASSES_noBG',None,'int')

    with open(input_file,'r') as f_obj:
        dir = os.path.dirname(input_file)
        files_list = f_obj.readlines()
        for file_name in files_list:
            file_name = file_name.strip()

            img_path = os.path.join(img_patch_dir,file_name+'.png')
            label_path = os.path.join(label_patch_dir,file_name+'.png')

            img_data = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            h_w = img_data.shape[:2]
            label_data = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
            if h_w != label_data.shape:
                b_diff = True
                basic.outputlogMessage('%s have different size in image (%s) and label (%s) patch'%(file_name,
                                                                        str(h_w),str(label_data.shape)))

            max_label = np.max(label_data)
            unique_value = np.unique(label_data)
            if max_label > num_classes_noBG:
                b_diff = True
                basic.outputlogMessage('%s: maximum pixel value (%d) in label images > num_class (%d)'%(file_name,max_label,num_classes_noBG))
            if len(unique_value) != num_classes_noBG + 1:
                b_diff = True
                basic.outputlogMessage('%s: the count of unique pixel value (%s) in label images not equals num_class (%d)' % (
                file_name, str(unique_value), num_classes_noBG))


    if b_diff is False:
        basic.outputlogMessage('all the patches are equal')



if __name__ == "__main__":
    usage = "usage: %prog [options] image_list "
    parser = OptionParser(usage=usage, version="1.0 2018-11-17")
    parser.description = 'Introduction: Check the label images after data augmentation '

    parser.add_option('-i','--image_dir',
                      action='store',dest='image_dir',default='split_images',
                      help="the folder of split image patches ")

    parser.add_option('-l','--label_dir',
                      action='store',dest='label_dir',default='split_labels',
                      help="the folder of split label patches")

    parser.add_option("-p", "--para_file",
                      action="store", dest="para_file",
                      help="the parameters file")


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    if options.para_file is None:
        print('error, parameter file is required')
        sys.exit(2)


    main(options, args)


