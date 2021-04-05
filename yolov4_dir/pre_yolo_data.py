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
            x, y, w, h = convert((width,height), (minX, maxX, minY, maxY))
            f_obj.writelines('%d %f %f %f %f\n'%(class_id, x, y, w, h))


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