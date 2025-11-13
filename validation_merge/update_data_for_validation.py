#!/usr/bin/env python
# Filename: update_data_for_validation.py 
"""
introduction: update the folder for validation.
"extract_grid_and_data.py" will try to extract sub-images for a given grid_vector and saved into a folder: validate_dir
If other validate results folder exists (e.g, result_dir1, result_dir2), then copies exists validated_*.json
before uploading these to web-based crowdsourcing system.

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 10 October, 2025
"""

import os,sys
from optparse import OptionParser

from numpy.ma.core import ravel

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)

import basic_src.basic as basic
import basic_src.io_function as io_function


def get_h3_folder(folder_path):
    folders = io_function.get_file_list_by_pattern(folder_path, '*')
    h3_folders = [item for item in folders if os.path.isdir(item) and len(os.path.basename(item)) == 15]
    return h3_folders

def copy_merge_validated_json(work_dir, h3_id, res_json_path):
    json_file_name = os.path.basename(res_json_path)
    save_json_path = os.path.join(work_dir,h3_id,json_file_name)
    if os.path.isfile(save_json_path):
        # merge info
        org_dict = io_function.read_dict_from_txt_json(save_json_path)
        new_dict = io_function.read_dict_from_txt_json(res_json_path)
        if org_dict['h3ID'] != new_dict['h3ID']:
            raise ValueError(f'The h3 ID in {save_json_path} and {res_json_path} is different')
        for key in new_dict.keys():
            org_dict[key] = new_dict[key]  # will replace the existing values
        io_function.save_dict_to_txt_json(save_json_path,org_dict)

    else:
        # copy the file
        io_function.copy_file_to_dst(res_json_path,save_json_path,b_verbose=True)


def copy_validation_res(work_dir, result_dir):
    # copy validation results from result_dir to the work_dir
    work_h3_folders = get_h3_folder(work_dir)
    res_h3_folders = get_h3_folder(result_dir)
    basic.outputlogMessage(f'To copy validation json files from {result_dir}')
    basic.outputlogMessage(f'work_dir: {len(work_h3_folders)} h3 folders, result_dir: {len(res_h3_folders)} h3 folders')

    h3_ids = [ os.path.basename(item) for item in work_h3_folders]
    for idx, h3_id in enumerate(h3_ids):
        h3_folder_in_res = os.path.join(result_dir,h3_id)
        # if the folder and folder exists, then copy and merge the validation information
        if os.path.isdir(h3_folder_in_res):
            validate_json = os.path.join(h3_folder_in_res,f'validated_{h3_id}.json')
            if os.path.isfile(validate_json):
                copy_merge_validated_json(work_dir,h3_id, validate_json)



def main(options, args):
    validate_dir = args[0]
    other_results_dir = [args[idx+1] for idx in range(len(args)-1)]
    basic.outputlogMessage(f'Data for validation are in {validate_dir}')
    basic.outputlogMessage(f'There are {len(other_results_dir)} folder potentially contain validation results, will copy them：')
    for tmp in other_results_dir:
        basic.outputlogMessage(tmp)

    for res_dir in other_results_dir:
        copy_validation_res(validate_dir, res_dir)



if __name__ == '__main__':

    usage = "usage: %prog [options] validate_dir result_dir1 result_dir2 ..."
    parser = OptionParser(usage=usage, version="1.0 2025-10-10")
    parser.description = 'Introduction: copy '

    # parser.add_option("-s", "--save_path",
    #                   action="store", dest="save_path",
    #                   help="the file path for saving the results")


    (options, args) = parser.parse_args()

    if len(sys.argv) < 2 or len(args) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)
