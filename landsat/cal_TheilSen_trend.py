#!/usr/bin/env python
# Filename: cal_TheilSen_trend
"""
introduction: Apply Theil-Sen Regression to landsat time series

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 12 June, 2019
"""

import sys,os
from optparse import OptionParser
import rasterio
import numpy as np

# import pandas as pd # read and write excel files

HOME = os.path.expanduser('~')
codes_dir2 = HOME + '/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)

import datetime
import matplotlib.pyplot as plt
import basic_src.RSImage as RSImage
from msi_landsat8 import get_band_names  # get_band_names(img_path)

import pandas as pd

from plot_landsat_timeseries import remove_nan_value
from plot_landsat_timeseries import get_date_string_list
from plot_landsat_timeseries import get_msi_file_list
from plot_landsat_timeseries import read_time_series



def main(options, args):

    msi_files = args






if __name__ == "__main__":
    usage = "usage: %prog [options] msi_file1 msi_file2 ..."
    parser = OptionParser(usage=usage, version="1.0 2019-4-14")
    parser.description = 'Introduction: calculate Theil-Sen Regression of landsat time series'

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
