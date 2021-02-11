#!/usr/bin/env python
# Filename: create_region_ini.py
"""
introduction:  create many region defined parameters by given a template and image paths

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 08 February, 2021
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

def create_new_region_defined_parafile(template_para_file, img_dir, area_remark=None):
    '''
    create a new region defined para file. Only defined the new images (did not change others)
    :param template_para_file:
    :param img_dir:
    :param area_remark:
    :return:
    '''
    io_function.is_file_exist(template_para_file)

    dir_base = os.path.basename(img_dir)
    date_strs = re.findall('\d{8}',dir_base)
    if len(date_strs) == 1:
        date = date_strs[0]
    else:
        date = 'unknown'

    new_para_file = io_function.get_name_by_adding_tail(template_para_file,date +'_'+ area_remark)
    new_para_file = os.path.basename(new_para_file) # save to current folder
    if os.path.isfile(new_para_file):
        raise IOError('%s already exists, please check or remove first')

    # copy the file
    io_function.copy_file_to_dst(template_para_file,new_para_file)

    if area_remark is not None:
        modify_parameter(new_para_file,'area_remark',area_remark)
    modify_parameter(new_para_file, 'input_image_dir', img_dir)
    modify_parameter(new_para_file, 'inf_image_dir', img_dir)

    tif_list = io_function.get_file_list_by_ext('.tif',img_dir, bsub_folder=False)
    if len(tif_list) < 1:
        raise ValueError('No tif in %s'%img_dir)
    if len(tif_list) == 1:
        modify_parameter(new_para_file, 'input_image_or_pattern', os.path.basename(tif_list[0]))
        modify_parameter(new_para_file, 'inf_image_or_pattern', os.path.basename(tif_list[0]))
    else:
        modify_parameter(new_para_file, 'input_image_or_pattern', '*.tif')
        modify_parameter(new_para_file, 'inf_image_or_pattern', '*.tif')

    print("modified and saved new parameter file: %s "%new_para_file)

    return new_para_file

def main(options, args):
    in_folder = args[0]
    template_ini = args[1]

    image_paths = io_function.get_file_list_by_ext('.tif',in_folder, bsub_folder=True)
    if len(image_paths) < 1:
        raise IOError('no tif files in %s'%in_folder)

    # get unique dir list
    img_dir_list = [ os.path.dirname(item) for item in image_paths ]
    img_dir_list = set(img_dir_list)

    region_ini_files_list = []
    for img_dir in img_dir_list:
        # copy template file
        out_ini = create_new_region_defined_parafile(template_ini,img_dir,options.area_remark)
        region_ini_files_list.append(out_ini)

    with open('region_ini_files.txt','a') as f_obj:
        for ini in region_ini_files_list:
            f_obj.writelines(ini + '\n')

    pass

if __name__ == '__main__':

    usage = "usage: %prog [options] image_folder template_ini "
    parser = OptionParser(usage=usage, version="1.0 2021-02-08")
    parser.description = 'Introduction: create a lot region-defined para file (ini) '

    parser.add_option("-r", "--area_remark",
                      action="store", dest="area_remark",
                      help="noted for the images")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 2:
        parser.print_help()
        sys.exit(2)
    main(options, args)