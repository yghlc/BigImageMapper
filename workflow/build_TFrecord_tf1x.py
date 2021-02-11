#!/usr/bin/env python
# Filename: build_TFrecord_tf1x.py 
"""
introduction: build TF record using tensorflow 1.x

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 29 January, 2021
"""

import os, sys
import time
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters

# the python with tensorflow 1.x installed
tf1x_python = 'python'

def call_build_TFrecord(build_script, para_file):

    command_string = tf1x_python + ' ' \
                     + build_script + ' ' \
                     + para_file
    res = os.system(command_string)
    if res != 0:
        sys.exit(1)

def main(options, args):

    para_file = args[0]
    network_setting_ini = parameters.get_string_parameters(para_file, 'network_setting_ini')

    global tf1x_python
    tf1x_python = parameters.get_file_path_parameters(network_setting_ini,'tf1x_python')

    build_script = os.path.join(code_dir, 'datasets', 'build_TFrecord.py')
    call_build_TFrecord(build_script,para_file)



if __name__ == '__main__':

    usage = "usage: %prog [options] para_file"
    parser = OptionParser(usage=usage, version="1.0 2021-01-21")
    parser.description = 'Introduction: build TF record files '

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)