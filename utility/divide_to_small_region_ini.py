#!/usr/bin/env python
# Filename: create_region_ini.py
"""
introduction:  divide a big region in (area*.ini) into many small regions for parallel prediction (image classification)

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 23 April, 2024
"""

import os, sys
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters

import basic_src.io_function as io_function


import re

def modify_parameter(para_file, para_name, new_value):
    parameters.write_Parameters_file(para_file,para_name,new_value)




def main(options, args):
    in_folder = args[0]
    template_ini = args[1]
    ext_name = options.extension

    image_paths = io_function.get_file_list_by_ext(ext_name,in_folder, bsub_folder=True)
    if len(image_paths) < 1:
        raise IOError('no tif files in %s'%in_folder)

    b_per_image_per_ini = options.b_per_image_per_ini
    region_ini_files_list = []
    if b_per_image_per_ini is False:
        # get unique dir list
        img_dir_list = [ os.path.dirname(item) for item in image_paths ]
        img_dir_list = set(img_dir_list)

        for img_dir in img_dir_list:
            # copy template file
            out_ini = create_new_region_defined_parafile(template_ini,img_dir,ext_name,area_name=options.area_name,
                                                           area_time=options.area_time, area_remark=options.area_remark)
            region_ini_files_list.append(out_ini)
    else:
        for img_path in image_paths:
            out_ini = create_region_parafile_for_one_image(template_ini,img_path,area_name=options.area_name,
                                                           area_time=options.area_time, area_remark=options.area_remark)
            region_ini_files_list.append(out_ini)


    with open('region_ini_files.txt','a') as f_obj:
        for ini in region_ini_files_list:
            f_obj.writelines(os.path.abspath(ini) + '\n')

    pass

if __name__ == '__main__':

    usage = "usage: %prog [options]  big_region_ini "
    parser = OptionParser(usage=usage, version="1.0 2024-04-23")
    parser.description = 'Introduction: divide a big region into many small regions (ini) '

    parser.add_option("-c", "--img_count",
                      action="store", dest="img_count", type=int,
                      help="image (grid) count per sub regions")

    parser.add_option("-s", "--save_dir",
                      action="store", dest="save_dir",
                      help="the folder to save the ini files of sub regions")


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 2:
        parser.print_help()
        sys.exit(2)
    main(options, args)