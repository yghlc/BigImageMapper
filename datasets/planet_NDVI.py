#!/usr/bin/env python
# Filename: planet_NDVI 
"""
introduction:

Ref: https://www.planet.com/docs/guides/quickstart-ndvi/

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

    # Load red and NIR bands - note all PlanetScope 4-band images have band order BGRN
    with rasterio.open(image_file) as src:
        band_red = src.read(3)

    with rasterio.open(image_file) as src:
        band_nir = src.read(4)

    # reflectance values was scaled by 10,000
    band_red = band_red/10000
    band_nir = band_nir/10000

    # Allow division by zero
    numpy.seterr(divide='ignore', invalid='ignore')

    # Calculate NDVI
    ndvi = (band_nir.astype(float) - band_red.astype(float)) / (band_nir + band_red)

    # Set spatial characteristics of the output object to mirror the input
    kwargs = src.meta
    kwargs.update(
        dtype=rasterio.float32,
        count=1)

    # Create the file
    with rasterio.open(output_file, 'w', **kwargs) as dst:
        dst.write_band(1, ndvi.astype(rasterio.float32))


if __name__ == "__main__":
    usage = "usage: %prog [options] input_image  output"
    parser = OptionParser(usage=usage, version="1.0 2018-7-15")
    parser.description = 'Introduction: calcuate NDVI of Planet images, ' \
                         'the input should be SR (surface Reflectance) product'

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)