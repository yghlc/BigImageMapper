#!/usr/bin/env python
# Filename: planet_NDVI 
"""
introduction: Normalized difference water index (NDWI)

Another is used to monitor changes related to water content in water bodies,
using green and NIR wavelengths, defined by McFeeters (1996):
Ref: https://en.wikipedia.org/wiki/Normalized_difference_water_index

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 15 July, 2018
"""


import sys,os
from optparse import OptionParser

import rasterio
import numpy


def main(options, args):
    image_file = args[0]
    output_file = args[1]

    print('input images:%s'%image_file)

    # Load green and NIR bands - note all PlanetScope 4-band images have band order BGRN
    with rasterio.open(image_file) as src:
        band_green = src.read(2)

    with rasterio.open(image_file) as src:
        band_nir = src.read(4)

    # # reflectance values was scaled by 10,000, not necessary
    # band_red = band_red/10000.0
    # band_nir = band_nir/10000.0

    # Allow division by zero
    numpy.seterr(divide='ignore', invalid='ignore')

    # Calculate NDVI
    ndvi = (band_green.astype(float) - band_nir.astype(float)) / (band_green + band_nir)

    # Set spatial characteristics of the output object to mirror the input
    kwargs = src.meta
    kwargs.update(
        dtype=rasterio.float32,
        count=1)

    # Create the file
    with rasterio.open(output_file, 'w', **kwargs) as dst:
        dst.write_band(1, ndvi.astype(rasterio.float32))

    print("save to %s"%output_file)

    # for display
    import matplotlib.pyplot as plt
    png = os.path.splitext(output_file)[0]+'.png'
    plt.imsave(png, ndvi, cmap=plt.cm.jet,vmin=-0.5, vmax=-0.1)  # set ndvi range [-1,1]

if __name__ == "__main__":
    usage = "usage: %prog [options] input_image  output"
    parser = OptionParser(usage=usage, version="1.0 2018-7-15")
    parser.description = 'Introduction: calcuate NDWI of Planet images, ' \
                         'the input should be SR (surface Reflectance) product'

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)