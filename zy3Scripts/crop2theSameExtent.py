#!/usr/bin/env python
# Filename: crop2theSameExtent 
"""
introduction: crop an image to the extent of the reference image

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 27 January, 2019
"""

import sys,os
from optparse import OptionParser

HOME = os.path.expanduser('~')

# path of DeeplabforRS
codes_dir2 = HOME + '/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)

# import basic_src.basic as basic
import basic_src.io_function as io_function
import basic_src.RSImageProcess as RSImageProcess

def main(options, args):
    input_image = args[0]
    base_image = args[1]

    if options.output is None:
        output = options.output
    else:
        output = io_function.get_name_by_adding_tail(input_image,'crop')

    RSImageProcess.subset_image_baseimage(output,input_image,base_image)


if __name__ == "__main__":
    usage = "usage: %prog [options] input_image base_image"
    parser = OptionParser(usage=usage, version="1.0 2019-1-4")
    parser.description = 'Introduction: crop an image to the extent of the reference image'

    parser.add_option("-o", "--output",
                      action="store", dest="output",
                      help="the output file path")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)
