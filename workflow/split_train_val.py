#!/usr/bin/env python
# Filename: split_train_val.py 
"""
introduction: split dataset to training and validation

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 19 January, 2021
"""

import os, sys


if __name__ == '__main__':

    print("%s : split data set into training and validation" % os.path.basename(sys.argv[0]))

    para_file = sys.argv[1]
    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    code_dir = os.path.join(os.path.dirname(sys.argv[0]), '..')
    sys.path.insert(0, code_dir)
    import parameters

    script = os.path.join(code_dir, 'datasets', 'train_test_split.py')

    training_data_per = parameters.get_digit_parameters_None_if_absence(para_file, 'training_data_per','float')
    train_sample_txt = parameters.get_string_parameters(para_file, 'training_sample_list_txt')
    val_sample_txt = parameters.get_string_parameters(para_file, 'validation_sample_list_txt')

    command_string = script + ' -t ' + str(training_data_per) + \
                     ' -s ' + train_sample_txt  + \
                     ' -v ' + val_sample_txt  + \
                     ' --shuffle ' + 'list/trainval.txt'
    os.system(command_string)