#!/usr/bin/env python
# Filename: multi_scale_analyze.py 
"""
introduction:

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

import vector_gpd

import h3

def merge_attributes(child_cells_path,attribute_name,parent_child_idx):
    child_values = vector_gpd.read_attribute_values_list(child_cells_path,attribute_name)
    parent_values = []
    for key in parent_child_idx.keys():
        values = [child_values[item] for item in parent_child_idx[key]]
        parent_values.append(sum(values))
    return parent_values


def convert_h3_cells_to_lower_scale(in_h3_cells,input_res,lower_res, lower_h3_cells):

    # link parent to children
    h3_id_child_list = vector_gpd.read_attribute_values_list(in_h3_cells,f'h3_id_{input_res}')
    print(f'read {len(h3_id_child_list)} child ids')
    h3_id_parent_list = vector_gpd.read_attribute_values_list(lower_h3_cells,f'h3_id_{lower_res}')
    print(f'read {len(h3_id_parent_list)} parent ids')

    # initiate the dict using h3_id_parent_list
    parent_child_idx = {}
    for h3_id in h3_id_parent_list:
        parent_child_idx[h3_id] = []

    for idx, child_id in enumerate(h3_id_child_list):
        parent_id = h3.cell_to_parent(child_id,lower_res)
        if parent_id in parent_child_idx.keys():
            parent_child_idx[parent_id].append(idx)
            # print(parent_id)
        else:
            print(f'warning, {parent_id} is not in the original parent id list')
            pass

    io_function.save_dict_to_txt_json('parent_child_idx_dict.txt',parent_child_idx)

    # add attributes
    attribute_name_list = ['comImg_C', 's2_occur']
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


def main():
    test_convert_h3_cells_to_lower_scale()
    pass


if __name__ == '__main__':
    main()
