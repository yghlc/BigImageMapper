#!/usr/bin/env python
# Filename: plot_cma_grid_data 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 05 June, 2019
"""

import sys,os
from optparse import OptionParser

import numpy as np
import rasterio

def read_grid_txt(file_path):
    '''
    read the txt file of grid data
    :param file_path:
    :return: a 2d array, lat, lon, cell size
    '''

    width = 0
    height = 0
    xll_lon = 0
    yll_lon =0
    res = 0
    img_array = None


    with open(file_path,'r') as f_obj:
        lines = f_obj.readlines()
        # read metadata
        width = int(lines[0].split()[1])
        height = int(lines[1].split()[1])
        xll_lon = float(lines[2].split()[1])
        yll_lon = float(lines[3].split()[1])
        res = float(lines[4].split()[1])
        no_data = float(lines[5].split()[1])

        # print(width,height,xll_lon,yll_lon,res)
        # img_array = np.zeros((height,width))
        row_values = []
        for idx in range(6, len(lines)):
            values = [ float(item) for item in lines[idx].split()]
            row_values.append(values)

        # read
        img_array = np.stack(row_values)
        # print(img_array.shape)

        return img_array,height,width,xll_lon,yll_lon,res,no_data


    # print(file_path)

def save_to_raster(array_2d,height,width,xll_lon,yll_lon,res,no_data,output):

    # save to an image
    with rasterio.Env():
        # Write an array as a raster band to a new 8-bit file. For
        # the new file's profile, we start with the profile of the source
        # profile = src.profile
        crs = rasterio.crs.CRS.from_epsg(4326)  # WGS 84,
        x_left_top = xll_lon
        y_left_top = yll_lon + res*height
        profile = {'driver': 'GTiff', 'nodata': no_data, 'width': width, 'height': height, 'count': 1,
                   'crs': crs,
                   'transform': (x_left_top, res, 0.0, y_left_top , 0.0, -res),
                   }

        # And then change the band count to 1, set the
        # dtype to uint8, and specify LZW compression.
        profile.update(
            dtype=rasterio.float32,
            count=1,
            compress='lzw')

        with rasterio.open(output, 'w', **profile) as dst:
            dst.write(array_2d.astype(rasterio.float32), 1)


def main(options, args):

    grid_txt = args[0]
    img_array, height, width, xll_lon, yll_lon, res, no_data =  read_grid_txt(grid_txt)

    save_to_raster(img_array, height, width, xll_lon, yll_lon, res, no_data, 'example.tif')


    pass


if __name__ == "__main__":
    usage = "usage: %prog [options] file_path"
    parser = OptionParser(usage=usage, version="1.0 2019-6-5")
    parser.description = 'Introduction: plot the grid data from CMA '

    parser.add_option("-o", "--output",
                      action="store", dest="output",
                      help="the output file path")

    parser.add_option("-d","--data_type",
                      action="store", dest="data_type",
                      help="data_tyep, including: pre, tem, gst (need to add more in the future)")


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)


    main(options, args)