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

from basic_src.xml_rw import OffsetMetaDataClass
import parameters

def main(options, args):
    ref_image = args[0]
    new_image = args[1]

    if options.output is not None:
        output = options.output
    else:
        output = io_function.get_name_by_adding_tail(new_image,'coreg')

    bkeepmidfile = True
    xml_path=os.path.splitext(output)[0]+'.xml'
    coreg_xml = OffsetMetaDataClass(xml_path)

    RSImageProcess.coregistration_siftGPU(ref_image,new_image,bkeepmidfile,coreg_xml)


if __name__ == "__main__":
    usage = "usage: %prog [options] ref_image new_image"
    parser = OptionParser(usage=usage, version="1.0 2019-1-29")
    parser.description = 'Introduction: co-registration of two images'

    parser.add_option("-o", "--output",
                      action="store", dest="output",
                      help="the output file path")

    parser.add_option("-p", "--para",
                      action="store", dest="para_file",
                      help="the parameters file")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    ## set parameters files
    if options.para_file is None:
        print('error, no parameters file')
        parser.print_help()
        sys.exit(2)
    else:
        parameters.set_saved_parafile_path(options.para_file)

    main(options, args)
