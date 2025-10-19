#!/usr/bin/env python
# Filename: update_data_for_validation.py 
"""
introduction: update the folder for validation.
"extract_grid_and_data.py" will try to extract sub-images for a given grid_vector and saved into a folder: validate_dir
If other validate results folder exists (e.g, result_dir1, result_dir2), then copies exists validated_*.json
before uplaoding these to web-based crowdsourcing system.

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 10 October, 2025
"""

import os,sys
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)

import basic_src.basic as basic
import basic_src.io_function as io_function



def main(options, args):
    validate_dir = args[0]
    other_results_dir = [args[idx+1] for idx in range(len(args)-1)]
    basic.outputlogMessage(f'Data for validation are in {validate_dir}')
    basic.outputlogMessage(f'There are {len(other_results_dir)} folder potentially contain validation results, will copy them：')
    for tmp in other_results_dir:
        basic.outputlogMessage(tmp)



if __name__ == '__main__':

    # sys.exit(0)
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
