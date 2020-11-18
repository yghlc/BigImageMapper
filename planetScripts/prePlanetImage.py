#!/usr/bin/env python
# Filename: prePlanetImage 
"""
introduction: Sharpen Planet images, apply for the 3 band images

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 17 June, 2018
"""

import sys,os
from optparse import OptionParser

import rasterio

import cv2
import numpy as np

def sharpen_planetImage(image_path,save_path):
    """
    perform shapening filter on an image
    :param image_path: the path of input image
    :param save_path:  the save path
    :return: True if successful, False otherwise
    """

    # image = cv2.imread(image_path)
    #     # print(image.shape)
    # cv2.imshow('Original', image)
    # Create our shapening kernel, it must equal to one eventually
    # kernel_sharpening = np.array([[-1,-1,-1],
    #                               [-1, 9,-1],
    #                               [-1,-1,-1]])

    with rasterio.open(image_path) as img_obj:
        # read the all bands (only have one band or three bands)
        indexes = img_obj.indexes
        profile = img_obj.profile
        if len(indexes) > 3:
            print('error, only support one band or three bands')
            return False
        if profile['dtype'] != 'uint8':
            print('error, only support dtype of unsigned 8 bit')
            return False

        img_data = img_obj.read(indexes)
        print('shape before transpose',img_data.shape)
        img_data = np.transpose(img_data,(1,2,0))
        print(img_data.shape)


    # Laplace
    kernel_sharpening = np.array([[0,-1,0],
                                  [-1, 5,-1],
                                  [0,-1,0]])

    # applying the sharpening kernel to the input image & displaying it.
    sharpened = cv2.filter2D(img_data, -1, kernel_sharpening)

    # cv2.imwrite(save_path,sharpened)
    # print("complete")

    # print("shape after filter", sharpened.shape)
    sharpened = np.transpose(sharpened,(2,0,1))
    # print("shape before save",sharpened.shape)
    profile.update(driver='GTiff')      # make sure the the output format is tif to avoid error: Writing through VRTSourcedRasterBand is not supported.
    with rasterio.open(save_path, "w", **profile) as dst:
        dst.write(sharpened.astype(rasterio.uint8), indexes)
        print('save result in %s' % save_path)


def main(options, args):

    input_path = args[0]
    output_path = args[1]

    sharpen_planetImage(input_path,output_path)

    pass


if __name__ == "__main__":
    usage = "usage: %prog [options] input_image output_image"
    parser = OptionParser(usage=usage, version="1.0 2018-4-10")
    parser.description = 'Introduction: Sharpen Planet images, only for one or three bands'

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)