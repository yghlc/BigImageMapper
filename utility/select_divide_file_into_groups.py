#!/usr/bin/env python
# Filename: select_divide_file_into_groups.py 
"""
introduction: select or divide many files (images) into many groups

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 30 July, 2025
"""


import os, sys
from optparse import OptionParser

sys.path.insert(0, os.path.expanduser('~/codes/PycharmProjects/DeeplabforRS'))
import basic_src.io_function as io_function
import geo_index_h3

import random


def random_divide_to_groups(file_list, num_each_group=100):
    """
    Randomly divides file_list into groups with up to num_each_group elements in each group.

    Args:
        file_list (list): List of file paths or items to divide.
        num_each_group (int): Number of items in each group.

    Returns:
        List[List]: List of groups (sublists). the last group may have fewer items
    """
    files = file_list.copy()
    random.shuffle(files)  # Shuffle in-place for randomness
    groups = [files[i:i + num_each_group] for i in range(0, len(files), num_each_group)]
    return groups

def copy_files(file_list, save_dir,group_id_str):
    for file in file_list:
        io_function.copyfiletodir(file, save_dir, b_verbose=False)

def move_files(file_list, save_dir,group_id_str):
    for file in file_list:
        io_function.movefiletodir(file, save_dir, b_verbose=False)

def organize_files_to_diff_folders(file_groups, save_dir='./random_groups', b_copy=False):
    if os.path.isdir(save_dir) is False:
        io_function.mkdir(save_dir)

    for g_id, group in enumerate(file_groups):
        g_id_str = f'g{g_id}'
        g_folder = os.path.join(save_dir,g_id_str)
        if os.path.isdir(g_folder) is False:
            io_function.mkdir(g_folder)
        if b_copy:
            copy_files(group,g_folder,g_id_str)
        else:
            move_files(group,g_folder,g_id_str)


def select_spatial_isolated_files(file_list, res_h3, thr_grid_dis):
    # to select files are there are no neighbours within a distance (thr_grid_dis)  as a h3 res (res_h3)
    h3_ids_list = [os.path.splitext(os.path.basename(item))[0] for item in  file_list]
    h3_ids_list = [item.split('_')[0]  for item in h3_ids_list] # for case like: 8e05100e9b8e317_1, remove "_1"
    # in default, we use shorted function
    h3_ids_list = [geo_index_h3.unshort_h3(item) for item in h3_ids_list]

    sel_ids, idx_list = geo_index_h3.select_isolated_files_h3_at_res(h3_ids_list, thr_grid_dis,res_h3 )

    sel_file_list = [file_list[item] for item in idx_list]
    print(f'Select {len(sel_file_list)} spatial isolated files from {len(file_list)} files')

    return sel_file_list

def main(options, args):
    data_folder = args[0]
    file_ext = options.file_ext
    num_each_group = options.num_each_group
    res_isolated_h3 = options.isolated_at_h3_res
    thr_neighbour_grid = options.neighbour_grid_k

    out_dir = data_folder+'_groups'
    if os.path.exists(out_dir):
        print(f'{out_dir} exists, please remove it if want to generate a new one')
        return

    file_list = io_function.get_file_list_by_ext(file_ext,data_folder,bsub_folder=True)

    if res_isolated_h3 is not None:
        file_list = select_spatial_isolated_files(file_list,res_isolated_h3,thr_neighbour_grid)

    file_groups = random_divide_to_groups(file_list,num_each_group=num_each_group)

    organize_files_to_diff_folders(file_groups, save_dir=out_dir, b_copy=True)



if __name__ == '__main__':
    usage = "usage: %prog [options] file_folder "
    parser = OptionParser(usage=usage, version="1.0 2025-7-30")
    parser.description = 'Introduction: select divide files into many group '

    parser.add_option("-e", "--file_ext",
                      action="store", dest="file_ext", type=str, default='.jpg',
                      help="the file extension for the selection")

    parser.add_option("-n", "--num_each_group",
                      action="store", dest="num_each_group", type=int, default=100,
                      help="the number of sample for selection in each group")

    parser.add_option("-r", "--isolated_at_h3_res",
                      action="store", dest="isolated_at_h3_res", type=int,
                      help="the h3 resolution to check if the file is isolated, default is None, then don't apply this")

    parser.add_option("-k", "--neighbour_grid_k",
                      action="store", dest="neighbour_grid_k", type=int, default=1,
                      help="if isolated_at_h3_res is set, k is the threshold for neighbouring ")




    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)
