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


def extract_classification_result_from_multi_sources(in_shp_list, save_path, extract_class_id=1, occurrence=4):
    '''
    extract the results which detected as the target multiple time during the multiple observation
    :param in_shp_list: file list for the multiple observation
    :param save_path: save path
    :param extract_class_id: the ID the target class
    :param occurrence: occurrence of the target during the multiple observation
    :return:
    '''

    if os.path.isfile(save_path):
        print(datetime.now(), '%s already exists, slip' % save_path)
        return

    print(datetime.now(), 'Input shape file:', in_shp_list)
    poly_class_ids_all = 'poly_class_ids_all.json'

    if os.path.isfile(poly_class_ids_all):
        print(datetime.now(), '%s already exists, read them directly'%poly_class_ids_all)
        poly_class_ids = io_function.read_dict_from_txt_json(poly_class_ids_all)
    else:
        # read
        poly_class_ids = {}
        for shp in in_shp_list:
            print(datetime.now(), 'reading %s'%shp)
            polyIDs = vector_gpd.read_attribute_values_list(shp,'polyID')
            preClassIDs = vector_gpd.read_attribute_values_list(shp, 'preClassID')
            _ = [poly_class_ids.setdefault(pid, []).append(c_id) for pid, c_id in zip(polyIDs,preClassIDs)]

        # save and organize them
        io_function.save_dict_to_txt_json('poly_class_ids_all.json',poly_class_ids)

    extract_class_id_results(in_shp_list[0], poly_class_ids, save_path, extract_class_id=extract_class_id, occurrence=occurrence)



def extract_class_id_results(shp_path, poly_class_ids, save_path, extract_class_id=1, occurrence = 4):
    '''
    extract the results which detected  as the target multiple time
    :param shp_path: a shape file contains points and "polyID"
    :param poly_class_ids: dict containing predicting results
    :param save_path: the save path for the results
    :param extract_class_id: the target id
    :param occurrence: occurrence time
    :return:
    '''
    save_json = 'poly_class_ids_id%d_occurrence%d.json' % (extract_class_id, occurrence)
    if os.path.isfile(save_json):
        print(datetime.now(),'warning, %s exists, read it directly'%save_json)
        sel_poly_class_ids = io_function.read_dict_from_txt_json(save_json)
        sel_poly_ids = list(sel_poly_class_ids.keys())
        print(datetime.now(), 'read %d results' % len(sel_poly_ids))
    else:
        print(datetime.now(), 'extract results for class: %d' % extract_class_id)
        sel_poly_ids = [ key for key in poly_class_ids.keys() if poly_class_ids[key].count(extract_class_id) >= occurrence ]
        print(datetime.now(), 'select %d results'%len(sel_poly_ids))
        # print(sel_poly_ids[:10])
        sel_poly_class_ids = {key: poly_class_ids[key] for key in sel_poly_ids}
        io_function.save_dict_to_txt_json(save_json, sel_poly_class_ids)

    # read and save shapefile
    save_shp = save_path
    sel_poly_ids_int = [int(item) for item in sel_poly_ids]
    vector_gpd.save_shapefile_subset_as_valueInlist(shp_path,save_shp,'polyID',sel_poly_ids_int)
    print(datetime.now(), 'save to %s and %s' % (save_json, save_shp))

def test_extract_class_id_results():
    shp_path = 'arctic_huang2023_620grids_s2_rgb_2023-predicted_classID.shp'
    poly_class_ids = io_function.read_dict_from_txt_json('poly_class_ids.json')
    extract_class_id_results(shp_path, poly_class_ids, extract_class_id=1, occurrence=7)


def main(options, args):

    res_shp_list = args
    save_path = options.save_path
    target_id = options.target_id
    min_occurrence = options.occurrence
    if save_path is None:
        save_path = 'extracted_results_classID%d_occurrence%d.shp'%(target_id,min_occurrence)


    extract_classification_result_from_multi_sources(res_shp_list, save_path,
                                                     extract_class_id = target_id,occurrence=min_occurrence)


if __name__ == '__main__':

    # test_extract_class_id_results()
    # sys.exit(0)

    usage = "usage: %prog [options] res_shp1 res_shp2 res_shp3 ...  "
    parser = OptionParser(usage=usage, version="1.0 2024-05-13")
    parser.description = 'Introduction: extract image classification results from multiple sources '

    parser.add_option("-s", "--save_path",
                      action="store", dest="save_path",
                      help="the file path for saving the results")

    parser.add_option("-i", "--target_id",
                      action="store", dest="target_id", type=int, default=1,
                      help="the class id want to save")

    parser.add_option("-m", "--occurrence",
                      action="store", dest="occurrence", type=int, default=4,
                      help="minimum of the target ID in multiple observations ")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)



    main(options, args)
