#!/usr/bin/env python
# Filename: only_get_sub_images_each_ini.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 26 December, 2021
"""

import os,sys

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters
import time

import basic_src.io_function as io_function

def main():
    # run in ~/Data/Arctic/canada_arctic/autoMapping/multiArea_sub_images on tesia
    ini_list = io_function.get_file_list_by_ext('.ini','./',bsub_folder=False)
    txt_list = io_function.get_file_list_by_pattern('./','area*.txt')
    for txt in txt_list:
        ini_s = io_function.read_list_from_txt(txt)
        ini_list.extend(ini_s)

    ini_list = [os.path.abspath(item) for item in ini_list]
    file_names = [ io_function.get_name_no_ext(item) for item in ini_list ]

    cur_dir = os.getcwd()

    # show
    [print(item) for item in ini_list]
    time.sleep(3)

    for name, area_ini in zip(file_names,ini_list):
        word_dir = os.path.join(cur_dir,name)
        io_function.mkdir(word_dir)
        os.chdir(word_dir)
        # copy and modify main_para.ini
        io_function.copyfiletodir(os.path.join(cur_dir,'main_para.ini'),'./',overwrite=True)
        io_function.copyfiletodir(os.path.join(cur_dir,'exe.sh'),'./',overwrite=True)

        parameters.write_Parameters_file('main_para.ini','training_regions',area_ini)

        # run exe.sh
        res = os.system('./exe.sh')
        if res !=0:
            print(res)
            sys.exit(1)

        os.chdir(cur_dir)





if __name__ == '__main__':
    main()