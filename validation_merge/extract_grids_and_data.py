#!/usr/bin/env python
# Filename: extract_grids_and_data.py 
"""
introduction: extract images, vector for each grid (cell) for validation

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 04 September, 2025
"""

import os,sys
import time
from optparse import OptionParser



code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)

import basic_src.io_function as io_function
import basic_src.basic as basic

import parameters

import numpy as np
import geopandas as gpd
from datetime import datetime
import re

import bim_utils
from utility.rename_subImages import rename_sub_images

def get_mapping_shp_raster_dict(pre_names, mapping_res_ini):
    mapping_shp_raster_dict = {}
    for p_name in pre_names:
        mapping_shp_raster_dict[p_name] = {}
        for v in ['-shp', '-dir', '-pattern']:
            ini_var_name = p_name + v
            if ini_var_name.endswith('-shp'):
                # shp_path = parameters.get_file_path_parameters(mapping_res_ini,ini_var_name)
                shp_path = parameters.get_file_path_parameters_None_if_absence(mapping_res_ini,ini_var_name)
                mapping_shp_raster_dict[p_name][ini_var_name] = shp_path
            if ini_var_name.endswith('-dir'):
                dir_path = parameters.get_directory_None_if_absence(mapping_res_ini, ini_var_name)
                mapping_shp_raster_dict[p_name][ini_var_name] = dir_path
            if ini_var_name.endswith('pattern'):
                pattern = parameters.get_string_parameters_None_if_absence(mapping_res_ini, ini_var_name)
                mapping_shp_raster_dict[p_name][ini_var_name] = pattern

    return mapping_shp_raster_dict

def get_h3_id_in_sub_images_f(file_path):
    ids = re.findall(r'id([0-9a-fA-F]+)_', os.path.basename(file_path))
    return ids[0]

def obtain_multi_data(grid_gpd, grid_vector_path,mapping_shp_raster_dict,out_dir, buffersize=10, process_num=4):

    dstnodata = 0
    rectangle_ext = True
    b_keep_org_file_name = True
    sub_image_format = 'GTIFF'
    img_file_extension = '.tif'
    # sub_image_format = 'PNG'
    # img_file_extension = '.png'

    # get sub-images for each raster, and the corresponding vectors
    for set_name in mapping_shp_raster_dict.keys():
        # extract sub-images
        # train_grids_shp,image_dir, buffersize,image_or_pattern,extract_img_dir,dstnodata,process_num,rectangle_ext,b_keep_org_file_name
        img_dir = mapping_shp_raster_dict[set_name][set_name+'-dir']
        img_pattern = mapping_shp_raster_dict[set_name][set_name+'-pattern']
        if img_dir is None or img_pattern is None:
            print(f'Warning: the image dir or pattern for {set_name} is None, skip it')
            continue
        sub_image_dir = os.path.join(out_dir, set_name)
        #extract sub-images
        bim_utils.extract_sub_images(grid_vector_path,img_dir,buffersize,img_pattern,sub_image_dir,dstnodata,process_num,rectangle_ext,b_keep_org_file_name,
                                     save_format=sub_image_format)

        # rename the sub-images
        sub_images_dir2 = os.path.join(sub_image_dir,'subImages')
        rename_sub_images(grid_vector_path, sub_images_dir2, img_file_extension, set_name, 'h3_id_8')
        print(datetime.now(), f'Extracted sub-images for {set_name}, saved in {sub_image_dir}')


    shp_gpd_dict = {}
    for set_name in mapping_shp_raster_dict.keys():
        shp_path = mapping_shp_raster_dict[set_name][set_name + '-shp']
        if shp_path is None:
            continue
        io_function.is_file_exist(shp_path)
        shp_gpd_dict[set_name] = gpd.read_file(shp_path)

    # save the grids in grid_gpd to geojson format
    vector_save_dir = out_dir
    save_h3_id_list = []
    for idx, row in grid_gpd.iterrows():
        h3_id = row["h3_id_8"]
        cell_vec_dir = os.path.join(vector_save_dir, h3_id)
        io_function.mkdir(cell_vec_dir)
        single_gdf = grid_gpd.iloc[[idx]]

        # save the cell grids
        cell_save_p = os.path.join(cell_vec_dir, f"h3_cell_{h3_id}.geojson")
        single_gdf.to_file(cell_save_p, driver="GeoJSON")
        save_h3_id_list.append(h3_id)

        # for shp dataset
        for set_name in shp_gpd_dict.keys():
            count = row[set_name+"_C"]
            if count < 1:
                continue
            shp_poly_save_p = os.path.join(cell_vec_dir, f"{set_name}_{h3_id}.geojson")
            # extract the corresponding vectors
            overlap_touch = gpd.sjoin(shp_gpd_dict[set_name], single_gdf, how='inner', predicate='intersects')
            # remove duplicated geometries in overlap_touch
            overlap_touch = overlap_touch.drop_duplicates(subset=['geometry'])  # only check geometry
            overlap_touch.to_file(shp_poly_save_p,driver="GeoJSON")

    # print(mapping_shp_raster_dict)

    # organize the sub-images
    sub_img_id_path_dict = {}
    for set_name in mapping_shp_raster_dict.keys():
        img_dir = mapping_shp_raster_dict[set_name][set_name + '-dir']
        if img_dir is None:
            continue
        sub_image_dir = os.path.join(out_dir, set_name,'subImages')
        sub_image_list = io_function.get_file_list_by_ext(img_file_extension,sub_image_dir,bsub_folder=False)
        for filename in sub_image_list:
            h3_id = get_h3_id_in_sub_images_f(filename)
            sub_img_id_path_dict.setdefault(h3_id, []).append(filename)
    for h3_id in sub_img_id_path_dict.keys():
        cell_dir = os.path.join(out_dir, h3_id)
        for img_path in sub_img_id_path_dict[h3_id]:
            io_function.movefiletodir(img_path,cell_dir,b_verbose=False)

    print(datetime.now(), f'Completed organizing sub-image')


def main(options, args):
    grid_path = args[0]
    out_dir = options.out_dir
    buffer_size= options.buffer_size

    mapping_res_ini = options.mapping_res_ini
    if mapping_res_ini is None:
        print('Please set "--mapping_res_ini"')
        return

    t0 = time.time()
    grid_gpd = gpd.read_file(grid_path)
    t1 = time.time()
    print(f'Loaded grid vector file, containing {len(grid_gpd)} cells, {len(grid_gpd.columns)} columns, cost {t1-t0} seconds')
    column_names = grid_gpd.columns.to_list()
    # remove "_A" (area) and "_C" (count), only keep these columns name end with "_A" or "_C"
    column_pre_names = [item.replace('_A','') for item in column_names if "_A" in item ]
    # column_pre_names = [item.replace('_C','') for item in column_pre_names]
    print('column names:', column_names)
    print('column_pre_names:', column_pre_names)

    mapping_shp_raster_dict = get_mapping_shp_raster_dict(column_pre_names, mapping_res_ini)
    io_function.save_dict_to_txt_json('mapping_shp_raster_dict.json',mapping_shp_raster_dict)

    if os.path.isdir(out_dir) is False:
        io_function.mkdir(out_dir)

    obtain_multi_data(grid_gpd,grid_path, mapping_shp_raster_dict, out_dir, buffersize=buffer_size)




if __name__ == '__main__':
    usage = "usage: %prog [options] grid_vector "
    parser = OptionParser(usage=usage, version="1.0 2025-9-4")
    parser.description = 'Introduction: extract multiple images and vector for each grid (cell) for validation '

    parser.add_option("-o", "--out_dir",
                      action="store", dest="out_dir",default="data_multi",
                      help="the output directory ")

    parser.add_option("-i", "--mapping_res_ini",
                      action="store", dest="mapping_res_ini",
                      help="the ini file contain mapping results and the corresponding rasters")

    parser.add_option("-b", "--buffer_size",
                      action="store", dest="buffer_size", type=float, default='50',
                      help="the buffer size (in meters) for extracting sub-images")

    # parser.add_option("-p", "--process_num",
    #                   action="store", dest="process_num",type=int, default=16,
    #                   help="the process number ")

    # parser.add_option("-b", "--using_bounding_box",
    #                   action="store_true", dest="using_bounding_box",default=False,
    #                   help="whether use the boudning boxes of polygons, this can avoid some invalid"
    #                        " polygons and be consistent with YOLO output")




    (options, args) = parser.parse_args()
    # print(options.no_label_image)
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)
