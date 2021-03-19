#!/usr/bin/env python
# Filename: scp_trained_model_to_curc.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 18 March, 2021
"""

import os, sys
code_dir = os.path.expanduser('~/codes/PycharmProjects/Landuse_DL')
sys.path.insert(0, code_dir)

import basic_src.io_function as io_function
import basic_src.basic as basic

import time

def get_remote_folder(remote_dir,folder_pattern):

    command_str = 'ssh $tesia_host "ls -d %s/%s"'%(remote_dir,folder_pattern)

    status, result = basic.getstatusoutput(command_str)
    # r_scan =$(ssh hlc @ 10.0.0.203 ls -d / home / hlc / Data / rock / Synchrotron / * _0${scan}_ *)
    # print(status)
    # print(result)
    dir_list = result.split()
    # print(dir_list)
    return dir_list

def main():

    basic.setlogfile('scp_log.txt')

    while True:
        # get remote dir
        basic.outputlogMessage('get remote folders')
        remote_folders = get_remote_folder(remote_dir, folder_pattern)
        basic.outputlogMessage("%d remote folders"%len(remote_folders))

        # get local dir
        folder_list = io_function.get_file_list_by_pattern(local_dir,folder_pattern)
        folder_list = [item for item in folder_list if os.path.isdir(item) ]
        folder_list.sort()
        basic.outputlogMessage("%d local folders" % len(folder_list))

        folder_name_list = [os.path.basename(item) for item in folder_list]

        for idx, r_folders in enumerate(remote_folders):
            folder_name = os.path.basename(r_folders)
            if folder_name in folder_name_list:
                continue

            basic.outputlogMessage('copy trained folder in %s'%folder_name)
            res = os.system('scp -r ${tesia_host}:%s %s/%s'%(remote_folders,local_dir,folder_name))

            if res !=0:
                sys.exit(1)

        # reomve incomplete folders
        for folder in folder_list:
            res_json = os.path.join(folder,'result.json')
            if os.path.isfile(res_json) and os.path.getsize(res_json) > 0:
                continue
            else:
                basic.outputlogMessage('remote incomplete folder %s'%os.path.basename(folder))
                io_function.delete_file_or_dir(folder)

        time.sleep(3600*5)  # wait five hours


    pass

if __name__ == '__main__':

    remote_dir = '/home/lihu9680/Data/Arctic/canada_arctic/autoMapping/ray_results/tune_dataAug_para_tesia'
    local_dir = '/home/lihu9680/Data/Arctic/canada_arctic/autoMapping/ray_results/tune_dataAug_para_tesia'

    folder_pattern = 'multiArea_deeplabv3P_?????'
    main()
    # get_remote_folder(remote_dir,folder_pattern)
    pass