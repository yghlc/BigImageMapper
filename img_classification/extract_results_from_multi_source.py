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
import basic_src.io_function as io_function
# import basic_src.map_projection as map_projection
import datasets.vector_gpd as vector_gpd


def extract_classification_result_from_multi_sources(in_shp_list, save_path, extract_class_id = 1):

    # read
    poly_class_ids = {}
    for shp in in_shp_list:
        polyIDs = vector_gpd.read_attribute_values_list(shp,'polyID')
        preClassIDs = vector_gpd.read_attribute_values_list(shp, 'preClassID')
        _ = [poly_class_ids.setdefault(pid, []).append(c_id) for pid, c_id in zip(polyIDs,preClassIDs)]


    # save and organize them
    io_function.save_dict_to_txt_json('poly_class_ids.json',poly_class_ids)



def main(options, args):

    res_shp_list = args
    save_path = options.save_path
    target_id = options.target_id
    if save_path is None:
        save_path = 'extracted_results_class_%d.shp'%target_id

    extract_classification_result_from_multi_sources(res_shp_list, save_path)


if __name__ == '__main__':
    usage = "usage: %prog [options] res_shp1 res_shp2 res_shp3 ...  "
    parser = OptionParser(usage=usage, version="1.0 2024-05-13")
    parser.description = 'Introduction: extract image classification results from multiple sources '

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    parser.add_option("-s", "--save_path",
                      action="store", dest="save_path",
                      help="the file path for saving the results")

    parser.add_option("-i", "--target_id",
                      action="store", dest="target_id", type=int, default=1,
                      help="the class id want to save")

    main(options, args)
