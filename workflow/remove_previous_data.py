#!/usr/bin/env python
# Filename: remove_previous_data.py 
"""
introduction: remove previous data to run again.

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 18 January, 2021
"""

import os,sys

if __name__ == '__main__':

    print("%s : remove previous data or results to run again" % os.path.basename(sys.argv[0]))

    para_file = sys.argv[1]

    if os.path.isfile(para_file) is False:
        raise IOError('File %s does not exists in current folder: %s' % (para_file, os.getcwd()))

    # print(os.path.abspath(sys.argv[0]))
    code_dir = os.path.join(os.path.dirname(sys.argv[0]), '..')
    sys.path.insert(0, code_dir)
    import parameters

    deeplabRS = parameters.get_directory_None_if_absence(para_file, 'deeplabRS_dir')
    # print(deeplabRS)
    sys.path.insert(0, deeplabRS)
    # print(sys.path)
    import basic_src.io_function as io_function

    subImage_dir = parameters.get_string_parameters_None_if_absence(para_file,'input_train_dir')
    subLabel_dir = parameters.get_string_parameters_None_if_absence(para_file,'input_label_dir')

    if os.path.isdir(subImage_dir):
        io_function.delete_file_or_dir(subImage_dir)
        print('remove %s' % subImage_dir)
    if os.path.isdir(subLabel_dir):
        io_function.delete_file_or_dir(subLabel_dir)
        print('remove %s' % subLabel_dir)








