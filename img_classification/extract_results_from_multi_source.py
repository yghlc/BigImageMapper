#!/usr/bin/env python
# Filename: extract_results_from_multi_source.py 
"""
introduction: extract image classification results from multiple sources

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 13 May, 2024
"""

import os,sys
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
# import parameters
# import basic_src.basic as basic
from datetime import datetime
import basic_src.io_function as io_function
# import basic_src.map_projection as map_projection
import datasets.vector_gpd as vector_gpd


def extract_classification_result_from_multi_sources(in_shp_list, save_path, extract_class_id=1):

    print('Input shape file:', in_shp_list)

    # read
    poly_class_ids = {}
    for shp in in_shp_list:
        print(datetime.now(), 'reading %s'%shp)
        polyIDs = vector_gpd.read_attribute_values_list(shp,'polyID')
        preClassIDs = vector_gpd.read_attribute_values_list(shp, 'preClassID')
        _ = [poly_class_ids.setdefault(pid, []).append(c_id) for pid, c_id in zip(polyIDs,preClassIDs)]



    # save and organize them
    io_function.save_dict_to_txt_json('poly_class_ids.json',poly_class_ids)


def extract_class_id_results(shp_path, poly_class_ids, extract_class_id=1, occurrence = 4):
    '''
    extract the results which detected  as the target multiple time
    :param shp_path: a shape file contains points and "polyID"
    :param poly_class_ids: dict containing predicting results
    :param extract_class_id: the target id
    :param occurrence: occurrence time
    :return:
    '''
    print(datetime.now(), 'extract results for class: %d' % extract_class_id)
    sel_poly_ids = [ key for key in poly_class_ids.keys() if poly_class_ids[key].count(extract_class_id) >= occurrence ]
    print(datetime.now(), 'select %d results'%len(sel_poly_ids))
    sel_poly_class_ids = {key: poly_class_ids[key] for key in sel_poly_ids}
    save_json = 'poly_class_ids_id%d_occurrence%d.json'%(extract_class_id,occurrence)
    io_function.save_dict_to_txt_json(save_json, sel_poly_class_ids)

    # read and save shapefile
    save_shp = 'poly_class_ids_id%d_occurrence%d.shp'%(extract_class_id,occurrence)
    polyID_list = vector_gpd.read_attribute_values_list(shp_path,'polyID')
    print(datetime.now(), 'read %d polyID '%len(polyID_list))
    sel_idxs = [ idx for idx, id in enumerate(polyID_list) if id in sel_poly_ids]
    print(datetime.now(), 'select %d polyID ' % len(sel_idxs))
    vector_gpd.save_shapefile_subset_as(sel_idxs,shp_path,save_shp)

    print(datetime.now(), 'save to %s and %s' % (save_json, save_shp))

def test_extract_class_id_results():
    shp_path = 'arctic_huang2023_620grids_s2_rgb_2023-predicted_classID.shp'
    poly_class_ids = io_function.read_dict_from_txt_json('poly_class_ids.json')
    extract_class_id_results(shp_path, poly_class_ids, extract_class_id=1, occurrence=7)


def main(options, args):

    res_shp_list = args
    save_path = options.save_path
    target_id = options.target_id
    if save_path is None:
        save_path = 'extracted_results_class_%d.shp'%target_id

    extract_classification_result_from_multi_sources(res_shp_list, save_path)


if __name__ == '__main__':

    test_extract_class_id_results()
    sys.exit(0)

    usage = "usage: %prog [options] res_shp1 res_shp2 res_shp3 ...  "
    parser = OptionParser(usage=usage, version="1.0 2024-05-13")
    parser.description = 'Introduction: extract image classification results from multiple sources '

    parser.add_option("-s", "--save_path",
                      action="store", dest="save_path",
                      help="the file path for saving the results")

    parser.add_option("-i", "--target_id",
                      action="store", dest="target_id", type=int, default=1,
                      help="the class id want to save")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)



    main(options, args)
