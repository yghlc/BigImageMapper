#!/usr/bin/env python
# Filename: multi_scale_analyze.py 
"""
introduction: convert and analyze results of H3 Grids at different scales

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 19 October, 2025
"""

import os,sys
from optparse import OptionParser
from datetime import datetime

deeplabforRS =  os.path.expanduser('~/codes/PycharmProjects/DeeplabforRS')
sys.path.insert(0, deeplabforRS)
import basic_src.io_function as io_function

import h3

import pandas as pd
import geopandas as gpd
import vector_gpd
import geo_index_h3 as geo_h3

def merge_attributes(child_cells_path,attribute_name,parent_child_idx):
    child_values = vector_gpd.read_attribute_values_list(child_cells_path,attribute_name)
    parent_values = []
    for key in parent_child_idx.keys():
        values = [child_values[item] for item in parent_child_idx[key]]
        parent_values.append(sum(values))
    return parent_values


def get_attribute_name_list(vector_path, attribute_suffix):
    # using geopandas to read all column names
    all_field_names = vector_gpd.read_attribute_name_list(vector_path)

    attribute_name_list = []
    for field in all_field_names:
        if field.endswith(attribute_suffix):
            attribute_name_list.append(field)
    return attribute_name_list

def convert_h3_cells_to_lower_scale(in_h3_cells,input_res,lower_res, lower_h3_cells):

    # link parent to children
    h3_id_child_list = vector_gpd.read_attribute_values_list(in_h3_cells,f'h3_id_{input_res}')
    print(f'read {len(h3_id_child_list)} child ids')
    h3_id_parent_list = vector_gpd.read_attribute_values_list(lower_h3_cells,f'h3_id_{lower_res}')
    print(f'read {len(h3_id_parent_list)} parent ids')

    # initiate the dict using h3_id_parent_list
    parent_child_idx = {}
    parent_child_h3_ids = {}
    for h3_id in h3_id_parent_list:
        parent_child_idx[h3_id] = []
        parent_child_h3_ids[h3_id] = []

    new_parent_ids = []

    for idx, child_id in enumerate(h3_id_child_list):
        parent_id = h3.cell_to_parent(child_id,lower_res)
        if parent_id in parent_child_idx.keys():
            parent_child_idx[parent_id].append(idx)
            parent_child_h3_ids[parent_id].append(child_id)
            # print(parent_id)
        else:
            print(f'warning, {parent_id} is not in the original parent id list, will create it')
            parent_child_idx[parent_id] = []
            parent_child_idx[parent_id].append(idx)
            new_parent_ids.append(parent_id)
            parent_child_h3_ids[parent_id] = []
            parent_child_h3_ids[parent_id].append(child_id)

    io_function.save_dict_to_txt_json('parent_child_idx_dict.txt',parent_child_idx)
    io_function.save_dict_to_txt_json('parent_child_h3_ids_dict.txt',parent_child_h3_ids)

    # create a new file if there are new parent ids
    if len(new_parent_ids) >0:
        lower_h3_cells_new = io_function.get_name_by_adding_tail(lower_h3_cells,'new')
        print(f'warning adding {len(new_parent_ids)} new parent h3 cells and save to {lower_h3_cells_new}')

        original_lower_cells_gpd = gpd.read_file(lower_h3_cells)  
        epsg_str = original_lower_cells_gpd.crs                 # to check 'EPSG:4326'
        print(f'original lower cells crs: {epsg_str}')

        new_cells_gpd = geo_h3.get_polygon_of_h3_cell(new_parent_ids, map_prj=epsg_str,h3_id_col_name=f'h3_id_{lower_res}')
        
        # merge original and new
        merged_gpd = gpd.GeoDataFrame(pd.concat([original_lower_cells_gpd, new_cells_gpd], ignore_index=True))
        merged_gpd.to_file(lower_h3_cells_new, driver=vector_gpd.guess_file_format_extension(lower_h3_cells_new))

        lower_h3_cells = lower_h3_cells_new

    # add attributes
    # attribute_name_list = ['comImg_C', 's2_occur']
    endwiths_suffix = ('_C','_A')
    attribute_name_list = get_attribute_name_list(in_h3_cells,endwiths_suffix)
    add_attributes = {}
    for att in attribute_name_list:
        values = merge_attributes(in_h3_cells,att,parent_child_idx)
        add_attributes[att] = values

    save_format = vector_gpd.guess_file_format_extension(lower_h3_cells)
    vector_gpd.add_attributes_to_shp(lower_h3_cells, add_attributes,format=save_format)





def test_convert_h3_cells_to_lower_scale():
    data_dir = os.path.expanduser('~/Data/rts_ArcticDEM_mapping/combine_mapping_results')
    child_h3_cells_vector = os.path.join(data_dir,'ygAlpha_res','h3_cells_select_by_rf_0.9.gpkg')
    child_res = 8
    parent_res = 3
    # parent_h3_cells_vector = os.path.join(data_dir,'h3_cells_for_merging_res4',
    #                     'h3_cells_res4_panArctic_s2_rgb_2024_object_detection_s2_exp5_post_1.shp')
    # parent_h3_cells_vector = os.path.join(data_dir,'h3_cells_for_merging_res4',
    # 'h3_cells_res4_panArctic_composited_image_2008-17_object_detection_comImg_exp3t2_post_1.shp')
    # parent_h3_cells_vector = os.path.join(data_dir,'h3_cells_for_merging_res4','for_debuging.shp')
    parent_h3_cells_vector = os.path.join(data_dir, 'h3_cells_panArctic_rts_res3.gpkg')

    # for_debuging.shp

    convert_h3_cells_to_lower_scale(child_h3_cells_vector,child_res,parent_res, parent_h3_cells_vector)


def main(options, args):
    # test_convert_h3_cells_to_lower_scale()
    child_h3_cells_vector = args[0]     # high_res_cells
    child_res = options.h3_high_res
    parent_h3_cells_vector = args[1]    # low_res_cells
    parent_res = options.h3_low_res

    convert_h3_cells_to_lower_scale(child_h3_cells_vector, child_res, parent_res, parent_h3_cells_vector)

    pass


if __name__ == '__main__':
    usage = "usage: %prog [options] high_res_cells  low_res_cells"
    parser = OptionParser(usage=usage, version="1.0 2025-12-6")
    parser.description = 'Introduction: convert h3 grid to lower resolution for analysis'

    parser.add_option("", "--h3_high_res",
                      action="store", dest="h3_high_res", type=int,default=8,
                      help="the resolution for the H3 grid cells at high resolution")

    parser.add_option("", "--h3_low_res",
                      action="store", dest="h3_low_res", type=int,default=3,
                      help="the resolution for the H3 grid cells at low resolution")


    (options, args) = parser.parse_args()
    # if len(sys.argv) < 2:
    #     parser.print_help()
    #     sys.exit(2)

    main(options, args)
