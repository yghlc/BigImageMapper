#!/usr/bin/env python
# Filename: training_img_augment 
"""
introduction: permaform data augmentation for both training images and label images

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 19 January, 2021
"""

import os,sys
import time

code_dir = os.path.join(os.path.dirname(sys.argv[0]), '..')
sys.path.insert(0, code_dir)
import parameters

def training_img_augment(para_file):

    print("start data augmentation")

    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s'%(para_file, os.getcwd()))

    # augscript = os.path.join(code_dir,'datasets','image_augment.py')

    img_ext = parameters.get_string_parameters_None_if_absence(para_file,'split_image_format')
    print("image format: %s"% img_ext)
    proc_num = parameters.get_digit_parameters(para_file, 'process_num', 'int')

    SECONDS=time.time()

    from datasets.image_augment import image_augment_main

    #augment training images
    print("image augmentation on image patches")
    img_list_aug_txt = 'list/images_including_aug.txt'
    # command_string = augscript + ' -p ' + para_file + ' -d ' + 'split_images' + ' -e ' + img_ext + ' -n ' + str(proc_num) + \
    #                  ' -o ' + 'split_images' + ' -l ' + img_list_aug_txt + ' ' + 'list/trainval.txt'
    # res = os.system(command_string)
    # if res!=0:
    #     sys.exit(1)
    image_augment_main(para_file,'list/trainval.txt',img_list_aug_txt,'split_images','split_images',img_ext,False,proc_num)

    #augment training lables
    print("image augmentation on label patches")
    # command_string = augscript + ' -p ' + para_file + ' -d ' + 'split_labels' + ' -e ' + img_ext + ' -n ' + str(proc_num) + \
    #                  ' -o ' + 'split_labels' + ' -l ' + img_list_aug_txt + ' ' + 'list/trainval.txt' + ' --is_ground_truth '
    #
    # res = os.system(command_string)
    # if res!=0:
    #     sys.exit(1)
    # save the result to the same file (redundant, they have the same filename)
    image_augment_main(para_file, 'list/trainval.txt', img_list_aug_txt, 'split_labels', 'split_labels', img_ext, True,proc_num)

    if os.path.isfile(img_list_aug_txt):
        os.system(' cp %s list/trainval.txt'%img_list_aug_txt)
        os.system(' cp %s list/val.txt'%img_list_aug_txt)
    else:
        print('list/images_including_aug.txt does not exist because no data augmentation strings')


    # output the number of image patches (ls may failed if there are a lot of files, so remove these two lines)
    # os.system('echo "count of class 0 ":$(ls split_images/*class_0*${img_ext} |wc -l) >> time_cost.txt')
    # os.system('echo "count of class 1 ":$(ls split_images/*class_1*${img_ext} |wc -l) >> time_cost.txt')

    duration= time.time() - SECONDS
    os.system('echo "$(date): time cost of data augmentation: %.2f seconds">>time_cost.txt'%duration)

if __name__ == '__main__':
    para_file = sys.argv[1]
    training_img_augment(para_file)


