#!/usr/bin/env python
# Filename: delete_bad_images.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 02 May, 2024
"""

import os,sys
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)

import datasets.raster_io as raster_io
import basic_src.io_function as io_function

def delete_images_low_quality(img_list, backup_folder):
    keep_image_label_list = []
    delete_image_label_list = []
    for image_path in img_list:
        valid_per, entropy = raster_io.get_valid_percent_shannon_entropy(image_path)  # base=10
        if valid_per > 60 and entropy >= 0.5:
            keep_image_label_list.append(image_path)
        else:
            delete_image_label_list.append(image_path)
            io_function.movefiletodir(image_path, backup_folder)


def main(options, args):

    img_folder = args[0]
    backup_folder = img_folder + '_delete'
    if os.path.isdir(backup_folder) is False:
        io_function.mkdir(backup_folder)
    img_list = io_function.get_file_list_by_ext('.tif',img_folder,bsub_folder=False)
    delete_images_low_quality(img_list,backup_folder)

if __name__ == '__main__':

    usage = "usage: %prog [options] image_folder "
    parser = OptionParser(usage=usage, version="1.0 2024-05-02")
    parser.description = 'Introduction: remove images with low quality  '

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)


