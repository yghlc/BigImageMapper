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

# import pandas as pd # read and write excel files

HOME = os.path.expanduser('~')
codes_dir2 = HOME + '/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)

from msi_landsat8 import get_band_names  # get_band_names(img_path)

def get_common_date_pathrow(red_band_name_list, green_band_name_list, blue_band_name_list):
    """

    :param red_band_name_list:
    :param green_band_name_list:
    :param blue_band_name_list:
    :return:
    """
    red_date_pathrow = [ '_'.join(item.split('_')[:2]) for item in red_band_name_list]
    green_date_pathrow = [ '_'.join(item.split('_')[:2]) for item in green_band_name_list]
    blue_date_pathrow = [ '_'.join(item.split('_')[:2]) for item in blue_band_name_list]

    # for item in red_date_pathrow:
    #     print(item)
    # print('************************')
    # get common bands (same date and pathrow)
    # the order of this list is random, how to avoid it? solution: sorted the output
    date_pathrow = sorted(set(red_date_pathrow) & set(green_date_pathrow) & set(blue_date_pathrow))
    # date_pathrow = set(red_date_pathrow) & set(green_date_pathrow)

    return date_pathrow

def read_one_band(img_path, band_name_list, date_pathrow):
    """
    read one band from the image
    :param img_path:
    :param date_pathrow:
    :return:
    """
    # get band name
    # print('start read ', date_pathrow)
    sel_band_name_idx = -1
    for idx, band_name in enumerate(band_name_list):
        if date_pathrow in band_name:
            sel_band_name_idx = idx
            break
    # print('reading ',date_pathrow)
    if sel_band_name_idx== -1 :
        raise ValueError("%s not in the file: %s"%(date_pathrow, img_path))

    with rasterio.open(img_path) as src:
        oneband_data = src.read(sel_band_name_idx + 1)      # this line is very slow
        # print('end read ', date_pathrow)
        return oneband_data


def main(options, args):

    brightness_file = args[0]
    greenness_file = args[1]
    wetness_file = args[2]

    b_band_name_list = get_band_names(brightness_file)
    g_band_name_list = get_band_names(greenness_file)
    w_band_name_list = get_band_names(wetness_file)

    # get common band with same date and path_row
    com_date_pathrows = get_common_date_pathrow(b_band_name_list, g_band_name_list,w_band_name_list)
    # print(com_date_pathrows)

    mean_list = []
    max_list = []
    min_list = []
    save_date_pathrow = []

    for idx, date_pathrow in enumerate(com_date_pathrows):

        # if idx < 30:
        #     continue
        # if idx > 31:
        #     break

        # read RGB bands
        print(date_pathrow)
        save_date_pathrow.append(date_pathrow)

        red_band = read_one_band(brightness_file, b_band_name_list, date_pathrow)
        green_band = read_one_band(greenness_file, g_band_name_list, date_pathrow)
        blue_band = read_one_band(wetness_file, w_band_name_list, date_pathrow)

        red_band = np.nan_to_num(red_band)
        green_band = np.nan_to_num(green_band)
        blue_band = np.nan_to_num(blue_band)

        ## save to file
        ## Set spatial characteristics of the output object to mirror the input
        # ref_src = rasterio.open(brightness_file)
        # kwargs = ref_src.meta
        # kwargs.update(
        #     dtype=rasterio.float32,
        #     count=3)
        # with rasterio.open('RGB_' + date_pathrow + '.tif', 'w', **kwargs) as dst:
        #     dst.write_band(1, red_band.astype(rasterio.float32))
        #     dst.write_band(2, green_band.astype(rasterio.float32))
        #     dst.write_band(3, blue_band.astype(rasterio.float32))

        # output max, min, mean of each bands
        mean_list.append([np.mean(red_band,axis=None), np.mean(green_band,axis=None), np.mean(blue_band,axis=None)]) #
        max_list.append([np.max(red_band),np.max(green_band), np.max(blue_band)])
        min_list.append([np.min(red_band), np.min(green_band), np.min(blue_band)])

        # print(idx)
        # break
        # for test
        # if idx > 0:
        #     break

    # print(mean_list)
    # print(max_list)
    # print(min_list)

    import csv
    row = ['date_pathrow', 'red_mean', 'green_mean','blue_mean','red_max', 'green_max','blue_max',
           'red_min', 'green_min','blue_min']
    with open('mean_max_min.csv', 'w') as csv_obj:
        wr = csv.writer(csv_obj) #, quoting=csv.QUOTE_ALL
        wr.writerow(row)

        for date_p, mean_s, max_s, min_s  in zip(save_date_pathrow,mean_list,max_list,min_list):
            wr.writerow([date_p,mean_s[0],mean_s[1],mean_s[2], max_s[0], max_s[1], max_s[2], min_s[0], min_s[1],min_s[2] ])


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
