#!/usr/bin/env python
# Filename: display_webpage.py 
"""
introduction: to display many tiny images and their labels or description in a html file (open in browser)

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 03 July, 2025
"""

import os,sys
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
# import parameters
import basic_src.basic as basic
import basic_src.io_function as io_function

def convert_tif_to_a_png_file(input_tif, save_path):
    cmd_str = f'gdal_translate -of PNG {input_tif} {save_path}'
    basic.os_system_exit_code(cmd_str)

def get_new_json_for_html(input_json, save_img_dir='display_PNG', save_json='display.json'):
    img_description_dict = io_function.read_dict_from_txt_json(input_json)
    if os.path.isdir(save_img_dir) is False:
        io_function.mkdir(save_img_dir)
    save_dict = {}
    for img_path in img_description_dict.keys():
        png_path = os.path.join(save_img_dir,io_function.get_name_no_ext(img_path) + '.png')
        convert_tif_to_a_png_file(img_path,png_path)
        save_dict[png_path] = img_description_dict[img_path]
    io_function.save_dict_to_txt_json(save_json,save_dict)


def main(options, args):
    get_new_json_for_html(args[0])

    pass


if __name__ == '__main__':
    usage = "usage: %prog [options] image_description.json "
    parser = OptionParser(usage=usage, version="1.0 2024-04-26")
    parser.description = 'Introduction: extract sub-images and sub-labels '

    parser.add_option("-t", "--template_html",
                      action="store", dest="template_html", default='image_description.html',
                      help="the template of the html file")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)
