#!/usr/bin/env python
# Filename: compose_RGB 
"""
introduction: Compose RGB images using Brightness, Greenness, and wetness

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 27 March, 2019
"""

import sys,os
from optparse import OptionParser
import rasterio
import numpy as np

HOME = os.path.expanduser('~')
codes_dir2 = HOME + '/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)

from msi_landsat8 import get_band_names  # get_band_names(img_path)

def main(options, args):

    brightness_file = args[0]
    greenness_file = args[1]
    wetness_file = args[2]

    band_name = get_band_names(brightness_file)
    print(band_name)





if __name__ == "__main__":
    usage = "usage: %prog [options] brightness_file greenness_file wetness "
    parser = OptionParser(usage=usage, version="1.0 2019-3-27")
    parser.description = 'Introduction: Compose RGB images using Brightness, Greenness, and wetness'

    parser.add_option("-o", "--output",
                      action="store", dest="output",
                      help="the output file path")

    # parser.add_option("-p", "--para",
    #                   action="store", dest="para_file",
    #                   help="the parameters file")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    ## set parameters files
    # if options.para_file is None:
    #     print('error, no parameters file')
    #     parser.print_help()
    #     sys.exit(2)
    # else:
    #     parameters.set_saved_parafile_path(options.para_file)

    main(options, args)
