#!/usr/bin/env python
# Filename: add_img_classify_result_to_vector.py
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 21 August, 2025
"""

import os,sys
from optparse import OptionParser


code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters
import basic_src.basic as basic
import basic_src.io_function as io_function
import datasets.vector_gpd as vector_gpd


def add_classified_results_to_vector(vector_path, sub_images_dir, column_name, res_label=1, file_ext='.tif', save_path=None):

    sub_image_list = io_function.get_file_list_by_ext(file_ext,sub_images_dir,bsub_folder=False)
    if len(sub_image_list) < 1:
        raise IOError(f'No sub-iamges in {sub_images_dir}, with file extension: {file_ext}')

    print(f'Found {len(sub_image_list)} sub-images in {sub_images_dir}')

    index_list = [ io_function.get_index_from_filename(item) for item in  sub_image_list]

    if vector_gpd.is_field_name_in_shp(vector_path, column_name):
        column_values = vector_gpd.read_attribute_values_list(vector_path,column_name)
    else:
        poly_list = vector_gpd.read_polygons_gpd(vector_path,b_fix_invalid_polygon=False)
        column_values = [-1] * len(poly_list)

    # update the values
    for idx in index_list:
        column_values[idx] = res_label
    save_dict = {column_name: column_values}

    if save_path is not None:
        save_format = vector_gpd.guess_file_format_extension(save_path)
    else:
        save_format = vector_gpd.guess_file_format_extension(vector_path)

    vector_gpd.add_attributes_to_shp(vector_path,save_dict,save_as=save_path, format=save_format)
    print(f'add or updated {len(index_list)} values to column: {column_name}')


def main(options, args):
    in_vector = args[0]
    sub_image_dir = args[1]
    io_function.is_folder_exist(sub_image_dir)
    io_function.is_file_exist(in_vector)

    column_name = options.column_name
    label_int_value = options.label_int_value
    save_path = options.save_as_path

    add_classified_results_to_vector(in_vector,sub_image_dir,column_name,res_label=label_int_value,save_path=save_path)



if __name__ == '__main__':
    usage = "usage: %prog [options] vector_file sub_img_dir "
    parser = OptionParser(usage=usage, version="1.0 2025-08-21")
    parser.description = 'Introduction: add image classified results (sub-images) to a vector file '

    parser.add_option("-c", "--column_name",
                      action="store", dest="column_name",default='class_int',
                      help="the column name to save the result")

    parser.add_option("-l", "--label_int_value",
                      action="store", dest="label_int_value",type=int, default=1,
                      help="the label for the sub-images in the folder")

    parser.add_option("-s", "--save_as_path",
                      action="store", dest="save_as_path",
                      help="if set, will save the vector to a new file")


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)


    main(options, args)
