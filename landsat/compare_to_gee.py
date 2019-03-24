#!/usr/bin/env python
# Filename:
"""
introduction: Compare the NDVI or other file computed locally to the one from Google Earth Engine.
for double-check purpose


authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 24 March, 2019
"""

import sys,os
from optparse import OptionParser
import rasterio

import numpy as np

HOME = os.path.expanduser('~')
codes_dir2 = HOME + '/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)


import basic_src.basic as basic


from basic_src.RSImage import RSImageclass


def get_band_names(img_path):
    '''
    get the all the band names (description) in this raster
    :param img_path:
    :return:
    '''
    rs_obj = RSImageclass()
    rs_obj.open(img_path)
    names = rs_obj.Getband_names()
    return names

    ##RasterBand.SetDescription(BandName) # This sets the band name!

def diff_bands(file_lcoal, file_gee):
    '''
    compare the bands of ndvi or other msi
    :param file_lcoal:
    :param file_gee:
    :return:
    '''

    band_names_local = get_band_names(file_lcoal)
    band_names_gee = get_band_names(file_gee)

    com_bands = set(band_names_local) & set(band_names_gee)
    basic.outputlogMessage('Common bands: ' + str(com_bands))

    # compare the bands with the same name
    for band_name in com_bands:

        local_src = rasterio.open(file_lcoal)
        band_local = local_src.read(band_names_local.index(band_name) + 1)

        gee_src = rasterio.open(file_gee)
        band_gee = gee_src.read(band_names_gee.index(band_name) + 1)

        # dff_value = band_local - band_gee
        dff_value = np.nan_to_num(band_local) - np.nan_to_num(band_gee)

        diff_sum = np.sum(dff_value,axis=None)
        diff_mean = np.mean(dff_value,axis=None)
        diff_max = np.max(dff_value)
        diff_min = np.min(dff_value)

        # find and output the different pixels
        loc_row, loc_col = np.where(np.abs(dff_value) > 0)

        # compare
        if  loc_col.size == 0:
            basic.outputlogMessage('%s band is total the samle'%band_name)
        else:
            basic.outputlogMessage('%s band is different'%band_name)

            # print(np.sum(dff_value))
            basic.outputlogMessage('difference (local - gee): sum %.6f, mean %.6f, min %.6f, max %.6f' %
                                   (diff_sum, diff_mean, diff_max, diff_min))

            for x,y in zip(loc_col,loc_row):
                # use gdallocationinfo to check the values
                print(x,y)

            # save this band
            # Set spatial characteristics of the output object to mirror the input
            kwargs = gee_src.meta
            kwargs.update(
                dtype=rasterio.float32,
                count=3)
            with rasterio.open('diff_'+band_name+'.tif', 'w', **kwargs) as dst:
                dst.write_band(1, band_local.astype(rasterio.float32))
                dst.write_band(2, band_gee.astype(rasterio.float32))
                dst.write_band(3, dff_value.astype(rasterio.float32))


    return True




def main(options, args):

    msi_file_local = args[0]
    msi_file_gee = args[1]

    diff_bands(msi_file_local,msi_file_gee)


if __name__ == "__main__":
    usage = "usage: %prog [options] msi_local  msi_gee "
    parser = OptionParser(usage=usage, version="1.0 2019-3-24")
    parser.description = 'Introduction: Compare the NDVI or other file computed locally to the one from Google Earth Engine'

    # parser.add_option("-o", "--output",
    #                   action="store", dest="output",
    #                   help="the output file path")

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
