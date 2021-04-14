#!/usr/bin/env python
# Filename: copy_json_for_other_dem_files 
"""
introduction: delineate thaw slumps on hillshade, then copy them to the relative DEM and slope derived from the same dem files.

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 13 April, 2021
"""


import os,sys

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)

import basic_src.io_function as io_function
import basic_src.basic as basic

import re

work_dir=os.path.expanduser('~/Data/Arctic/alaska/time_series_sub_images_191Polys')

def get_date_str_poly_num(file_name):
    date_str = re.findall('[0-9]{8}', file_name)[0]
    poly_num = re.findall('poly_\d+', file_name)[0]
    return date_str, poly_num


def main():

    hillshade_dir = os.path.join(work_dir,'hillshade_sub_images')
    dem_slope_8bit_dir = os.path.join(work_dir,'dem_slope_8bit_sub_images')
    dem_relative_8bit_dir = os.path.join(work_dir,'dem_relative_8bit_sub_images')
    other_dirs = [dem_slope_8bit_dir,dem_relative_8bit_dir]
    other_dirs_tifs = [ io_function.get_file_list_by_ext('.tif', o_dir, bsub_folder=True) for o_dir in  other_dirs]


    json_list = io_function.get_file_list_by_ext('.json', hillshade_dir, bsub_folder=True)
    json_base_list = [os.path.basename(item) for item in json_list]

    for json_path, base_name in zip(json_list, json_base_list):
        date_str, poly_num = get_date_str_poly_num(base_name)

        for tif_list in other_dirs_tifs:

            for tif in tif_list:
                name_noext = io_function.get_name_no_ext(tif)
                if date_str in name_noext and poly_num in name_noext:
                    # modify and save the json file
                    dst_path = os.path.join(os.path.dirname(tif), name_noext+'.json')
                    # io_function.copy_file_to_dst(json_path,dst_path)
                    data_dict = io_function.read_dict_from_txt_json(json_path)
                    data_dict['imagePath'] = os.path.basename(tif)
                    data_dict['imageData'] = None
                    io_function.save_dict_to_txt_json(dst_path, data_dict)
                    print('saving %s'%dst_path)

                    break

        pass


if __name__ == '__main__':
    main()
    pass