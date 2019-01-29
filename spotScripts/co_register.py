#!/usr/bin/env python
# Filename: crop2theSameExtent 
"""
introduction: co-registration of two images

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 29 January, 2019
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
import basic_src.geometryProcess as geometryProcess

from basic_src.xml_rw import OffsetMetaDataClass

def main(options, args):
    ref_image = args[0]
    new_image = args[1]

    if options.output is not None:
        output = options.output
    else:
        output = io_function.get_name_by_adding_tail(new_image,'coreg')

    bkeepmidfile = True
    coreg_xml = OffsetMetaDataClass()

    RSImageProcess.coregistration_siftGPU(ref_image,new_image,bkeepmidfile,coreg_xml)


if __name__ == "__main__":
    usage = "usage: %prog [options] ref_image new_image"
    parser = OptionParser(usage=usage, version="1.0 2019-1-29")
    parser.description = 'Introduction: co-registration of two images'

    parser.add_option("-o", "--output",
                      action="store", dest="output",
                      help="the output file path")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)
