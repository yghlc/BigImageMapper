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
        io_function.copyfiletodir(file, save_dir)

def move_files(file_list, save_dir,group_id_str):
    for file in file_list:
        io_function.movefiletodir(file, save_dir)

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


def main(options, args):
    data_folder = args[0]
    file_ext = options.file_ext
    num_each_group = options.num_each_group

    file_list = io_function.get_file_list_by_ext(file_ext,data_folder,bsub_folder=True)
    file_groups = random_divide_to_groups(file_list,num_each_group=num_each_group)

    out_dir = data_folder+'_groups'
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


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)
