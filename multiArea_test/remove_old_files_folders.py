#!/usr/bin/env python
# Filename: move_old_files_folders.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 08 March, 2021
"""
import os,sys
from optparse import OptionParser
# import difflib
import time
from datetime import datetime

code_dir = os.path.expanduser('~/codes/PycharmProjects/Landuse_DL')
sys.path.insert(0, code_dir)
import basic_src.io_function as io_function


time_hour_thr = 10

from tune_training_para_curc import remove_files

def main(options, args):
    root_dir = args[0]
    if os.path.isdir(root_dir) is False:
        raise ValueError('%s not exists'%root_dir)

    folder_pattern = options.folder_pattern

    checked_folders = []

    while True:

        folder_list = io_function.get_file_list_by_pattern(root_dir, folder_pattern)
        folder_list = [item for item in folder_list if os.path.isdir(item)]
        folder_list.sort()

        # ray may create a new folder if the previous one already exists
        dupli_foldes = io_function.get_file_list_by_pattern(root_dir, folder_pattern + '_????')
        dupli_foldes = [item for item in dupli_foldes if os.path.isdir(item)]
        dupli_foldes.sort()
        folder_list.extend(dupli_foldes)

        print(str(datetime.now()), 'start moving or removing files or folders\n')
        check_folder_count = 0
        for folder in folder_list:
            if folder in checked_folders:
                continue

            print('checking folder: %s' % folder)
            if io_function.check_file_or_dir_is_old(folder,time_hour_thr):
                print('%s is older than %f hours, will remove some files inside it'%(folder, time_hour_thr))
                remove_files(folder)
                checked_folders.append(folder)
                check_folder_count += 1

        print(str(datetime.now()), 'removing files in %d folders'%check_folder_count)
        time.sleep(60)  # wait

    pass


if __name__ == '__main__':
    usage = "usage: %prog [options] training_folder "
    parser = OptionParser(usage=usage, version="1.0 2021-03-13")
    parser.description = 'Introduction: collect parameters and training results (miou) '

    parser.add_option("-f", "--folder_pattern",
                      action="store", dest="folder_pattern",default='multiArea_deeplabv3P_?????',
                      help="the pattern of training folder")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)
    main(options, args)
