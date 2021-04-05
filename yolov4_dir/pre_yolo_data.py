#!/usr/bin/env python
# Filename: pre_yolo_data 
"""
introduction: conver training data for semantic to yolo objection.

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 04 April, 2021
"""

import os,sys
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import basic_src.io_function as io_function
import parameters
import time

import datasets.raster_io as raster_io
from datasets.get_boxes_label_images import get_boxes_from_label_image
from yoltv4Based.yolt_func import convert

from yoltv4Based.yolt_func import convert_reverse

def get_yolo_boxes_one_img(idx, total, image_path, label_path):
    print('to yolo box: %d/%d'%(idx+1, total))
    # a object : [ class_id,  minX, minY, maxX, maxY ]
    objects = get_boxes_from_label_image(label_path)
    save_object_txt = os.path.splitext(image_path)[0] + '.txt'
    height, width, count, dtype = raster_io.get_height_width_bandnum_dtype(label_path)

    with open(save_object_txt, 'w') as f_obj:
        for object in objects:
            class_id, minX, minY, maxX, maxY = object
            # in semantic, class_id 0 is background, yolo, class 0 is target, so minus 1
            class_id -= 1
            x, y, w, h = convert((width,height), (minX, maxX, minY, maxY))
            f_obj.writelines('%d %f %f %f %f\n'%(class_id, x, y, w, h))

def get_image_list(txt_dir,sample_txt,img_dir, img_ext):
    img_list = []
    with open(os.path.join(txt_dir,sample_txt), 'r') as f_obj:
        lines = [item.strip() for item in  f_obj.readlines()]
        for line in lines:
            img_list.append(os.path.join(img_dir, line + img_ext))
    return img_list

def image_label_to_yolo_format(para_file):

    print("Image labels (semantic segmentation) to YOLO object detection")

    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    img_ext = parameters.get_string_parameters_None_if_absence(para_file,'split_image_format')
    proc_num = parameters.get_digit_parameters(para_file, 'process_num', 'int')

    SECONDS = time.time()

    # get image and label path
    image_list = []
    label_list = []
    with open(os.path.join('list','trainval.txt'), 'r') as f_obj:
        lines = [item.strip() for item in  f_obj.readlines()]
        for line in lines:
            image_list.append(os.path.join('split_images', line + img_ext))
            label_list.append(os.path.join('split_labels', line + img_ext))


    # get boxes
    total_count = len(image_list)
    for idx, (img, label) in enumerate(zip(image_list,label_list)):
        get_yolo_boxes_one_img(idx, total_count, img, label)


    # write obj.data file
    train_sample_txt = parameters.get_string_parameters(para_file, 'training_sample_list_txt')
    val_sample_txt = parameters.get_string_parameters(para_file, 'validation_sample_list_txt')
    train_img_list = get_image_list('list',train_sample_txt,'split_images',img_ext)
    val_img_list = get_image_list('list',val_sample_txt,'split_images',img_ext)

    expr_name = parameters.get_string_parameters(para_file,'expr_name')
    object_names = parameters.get_string_list_parameters(para_file,'object_names')
    num_classes_noBG = parameters.get_digit_parameters_None_if_absence(para_file, 'NUM_CLASSES_noBG', 'int')
    io_function.mkdir('data')
    io_function.mkdir(expr_name)

    with open(os.path.join('data','obj.data'), 'w') as f_obj:
        f_obj.writelines('classes = %d'%num_classes_noBG + '\n')

        train_txt = os.path.join('data','train.txt')
        io_function.save_list_to_txt(train_txt,train_img_list)
        f_obj.writelines('train = %s'%train_txt+ '\n')

        val_txt = os.path.join('data','val.txt')
        io_function.save_list_to_txt(val_txt, val_img_list)
        f_obj.writelines('valid = %s' % val_txt + '\n')

        obj_name_txt = os.path.join('data','obj.names')
        io_function.save_list_to_txt(obj_name_txt,object_names)
        f_obj.writelines('names = %s' % obj_name_txt + '\n')

        f_obj.writelines('backup = %s'%expr_name + '\n')


    duration = time.time() - SECONDS
    os.system('echo "$(date): time cost of converting to yolo format: %.2f seconds">>time_cost.txt' % duration)

    pass



def main(options, args):
    para_file= args[0]
    image_label_to_yolo_format(para_file)


if __name__ == '__main__':
    usage = "usage: %prog [options] para_file "
    parser = OptionParser(usage=usage, version="1.0 2022-04-04")
    parser.description = 'Introduction: convert split images and labels to yolo format (objection) '

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)