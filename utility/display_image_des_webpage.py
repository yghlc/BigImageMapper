#!/usr/bin/env python
# Filename: display_image_des_webpage.py
"""
introduction: to display many tiny images and their labels or description in a html file (open in browser)

# in the same folder of the html file, start "python3 -m http.server", then open the html file in: http://127.0.0.1:8000/image_description.html
# "image_description.html" need to change if the filename changed.

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 03 July, 2025
"""

import os,sys
from optparse import OptionParser

from tqdm import tqdm

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
# import parameters
import basic_src.basic as basic
import basic_src.io_function as io_function
import datasets.raster_io as raster_io

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

def multi_image_to_png_files(image_list, save_dir):
    if os.path.isdir(save_dir) is False:
        io_function.mkdir(save_dir)
    save_path_list = []
    for idx, img_path in enumerate(image_list):
        png_path = os.path.join(save_dir, io_function.get_name_no_ext(img_path) + '.png')
        convert_tif_to_a_png_file(img_path, png_path)
        save_path_list.append(png_path)
    return save_path_list

def files_for_display_similarity_matrix(reference_image_txt, search_image_txt, similarity_matrix_txt=None):
    #  const referenceImagesFile = 'reference_images.txt';
    #  const searchImagesFile = 'search_images.txt';
    #  const similarityMatrixFile = 'similarity_matrix.txt';
    # see html file (display_image_similarity.html) in ~/Data/slump_demdiff_classify/clip_classify/image_text_cluster on ygAlpha

    ref_img_list = io_function.read_list_from_txt(reference_image_txt)
    search_img_list = io_function.read_list_from_txt(search_image_txt)

    ref_png_list = multi_image_to_png_files(ref_img_list, 'reference_images_PNG')
    io_function.save_list_to_txt('reference_images.txt',ref_png_list)

    search_png_list = multi_image_to_png_files(search_img_list, 'search_image_images_PNG')
    io_function.save_list_to_txt('search_images.txt',search_png_list)


def display_images_values_s2(image_value_json, rgb_bands=[1,2,3], img_dir=None, save_img_dir='display_PNG', save_json='display.json',b_sorted=True):
    # convert images to 8 bit RGB and values to their description, then save to json file,
    # for displaying many images in a webpage
    img_value_dict = io_function.read_dict_from_txt_json(image_value_json)
    if os.path.isdir(save_img_dir) is False:
        io_function.mkdir(save_img_dir)

    # each values contains two elements: [valid_percent, entropy ]
    # Sort the dictionary by its values (ascending)
    if b_sorted:
        img_value_dict = dict(sorted(img_value_dict.items(), key=lambda item: item[1][1]))

    save_dict = {}
    for img_path in tqdm(img_value_dict.keys(), desc='Images to RGB PNG'):
        png_path = os.path.join(save_img_dir, io_function.get_name_no_ext(img_path) + '.png')
        if img_dir is not None:
            img_path_2 = os.path.join(img_dir,img_path)
        else:
            img_path_2 = img_path

        # for sentinel-2 images
        raster_io.convert_images_to_rgb_8bit_np(img_path_2,save_path=png_path,rgb_bands=rgb_bands,sr_min=0,sr_max=1600,
                                                nodata=0,format='PNG',verbose=False)
        valid_percent = img_value_dict[img_path][0]
        entropy = img_value_dict[img_path][1]
        save_dict[png_path] = f"v_perc={valid_percent:.1f}%, entropy={entropy:.3f}" # str(img_value_dict[img_path])

    io_function.save_dict_to_txt_json(save_json, save_dict)




def main(options, args):

    # for the image description
    if len(args)==1 and args[0].endswith('.json'):
        if options.to_rgb_8bit:
            display_images_values_s2(args[0], rgb_bands=[1, 2, 3], img_dir=options.image_dir, save_img_dir='display_PNG',
                                     save_json='display.json', b_sorted=True)
        else:
            get_new_json_for_html(args[0])
        return

    # for similarity matrix
    if len(args) == 2 and args[0].endswith('.txt') and args[1].endswith('.txt'):
        files_for_display_similarity_matrix(args[0], args[1])
        return

    print('Do nothing, please check the input')


if __name__ == '__main__':
    usage = "usage: %prog [options] image_description.json OR ref_image_list.txt search_image_list.txt "
    parser = OptionParser(usage=usage, version="1.0 2025-07-03")
    parser.description = 'Introduction: display many images and corresponding lable/description in a webpage '

    parser.add_option("-t", "--template_html",
                      action="store", dest="template_html", default='image_description.html',
                      help="the template of the html file")

    parser.add_option("-d", "--image_dir",
                      action="store", dest="image_dir",
                      help="the root directory of all images")

    parser.add_option("", "--to_rgb_8bit",
                      action="store_true", dest="to_rgb_8bit",default=False,
                      help="indicate if the input is sentinel-2 and need to convert to 8 bit RGB")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)
