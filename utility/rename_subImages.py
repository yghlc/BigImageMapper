#!/usr/bin/env python
# Filename: rename_subImages.py
"""
introduction: to rename the file names of sub-images extracted by "get_subImages.py"

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 19 August, 2025
"""

import os,sys
from optparse import OptionParser

deeplabforRS =  os.path.expanduser('~/codes/PycharmProjects/DeeplabforRS')
sys.path.insert(0, deeplabforRS)
import vector_gpd
import basic_src.io_function as io_function
import basic_src.basic as basic


def rename_sub_images(vector_for_extracting, img_dir,img_file_extension, pre_name, id_column, out_dir=None):

    b_copied = False
    if out_dir is not None:
        io_function.mkdir(out_dir)
        b_copied =True

    if vector_gpd.is_field_name_in_shp(vector_for_extracting,id_column) is False:
        raise ValueError(f'Column: {id_column} is not in {vector_for_extracting}')

    id_list = vector_gpd.read_attribute_values_list(vector_for_extracting, id_column)
    file_list = io_function.get_file_list_by_ext(img_file_extension,img_dir,bsub_folder=False)
    if len(file_list) < 1:
        basic.outputlogMessage(f'No images in {img_file_extension}, with file extension: {img_file_extension}')
        return

    if len(id_list) < len(file_list):
        raise ValueError(f'The count {len(id_list)} of IDs in {vector_for_extracting} is less than total image count: {len(file_list)}')

    for img in file_list:
        img_idx = io_function.get_index_from_filename(os.path.basename(img))
        img_id = id_list[img_idx]
        new_file_name = f"{pre_name}_id{img_id}_{img_idx}{img_file_extension}"
        if b_copied:
            new_path = os.path.join(out_dir,new_file_name)
            io_function.copy_file_to_dst(img, new_path,overwrite=False,b_verbose=True)
        else:
            new_path = os.path.join(os.path.dirname(img), new_file_name)
            io_function.move_file_to_dst(img,new_path,b_verbose=True)


def main(options, args):
    input_vector = args[0]
    img_dir = args[1]
    out_dir = options.out_dir
    id_column = options.id_column
    pre_name = options.pre_name
    image_ext = options.image_ext
    rename_sub_images(input_vector, img_dir, image_ext, pre_name, id_column, out_dir=out_dir)
    pass


if __name__ == "__main__":
    usage = "usage: %prog [options] vector_file image_dir "
    parser = OptionParser(usage=usage, version="1.0 2025-8-19")
    parser.description = 'Introduction: renames file name of sub-images  '

    parser.add_option("-o", "--out_dir",
                      action="store", dest="out_dir",
                      help="if this is set, files will be copied and save to another folder")

    parser.add_option("-i", "--id_column",
                      action="store", dest="id_column",default='RowCol_id',
                      help="the name of unique ID column")

    parser.add_option("-p", "--pre_name",
                      action="store", dest="pre_name", default='Img',
                      help="the prename for the file")

    parser.add_option("-e", "--image_ext",
                      action="store", dest="image_ext",default = '.tif',
                      help="the image pattern of the image file")


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)
