#!/usr/bin/env python
# Filename: image_classify_rnn 
"""
introduction: classification multi-spectral remote sensing images using RNN. This is pixel-based classification

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 10 April, 2018
"""


import sys,os
from optparse import OptionParser

import numpy as np
import random

import rasterio

def read_oneband_image_to_1dArray(image_path):

    if os.path.isfile(image_path) is False:
        print("error, file not exist: " + image_path)
        return None

    with rasterio.open(image_path) as img_obj:
        # read the all bands (only have one band)
        indexes = img_obj.indexes
        if len(indexes) != 1:
            print('error, only support one band')
            return None

        data = img_obj.read(indexes)

        data_1d = data.flatten()  # convert to one 1d, row first.

        return data_1d


def read_multiband_image_to_2dArray(image_path):
    """

    :param image_path:
    :return: 2d Array (bands, number of pixels)
    """

    if os.path.isfile(image_path) is False:
        print("error, file not exist: " + image_path)
        return None

    with rasterio.open(image_path) as img_obj:
        # read the all bands (only have one band)
        indexes = img_obj.indexes

        data = img_obj.read(indexes)
        band,height, width = data.shape
        # print(data.shape)
        data = np.transpose(data,(1,2,0))  # this is necessary before reshape, or the arrary is wrong
        data_2d = data.reshape(height*width,band)  # row first
        return data_2d

def split_data(x_all, y_all, test_percent=0.01):
    """
    split the data to train set and test set
    :param x_all: all the x : (count,n_features)
    :param y_all: all the y : (count,1)
    :param percent: the percent of the test set [0,1]
    :return: (x_train, y_train), (x_test, y_test)
    """

    total_count = x_all.shape[0]
    test_count = int(total_count*test_percent)

    # random select the test sample
    test_index = np.random.randint(0,total_count,size=test_count)

    x_test = x_all[test_index]
    y_test = y_all[test_index]

    x_train = np.delete(x_all,test_index,axis=0)
    y_train = np.delete(y_all,test_index,axis=0)

    return (x_train, y_train), (x_test, y_test)


def main(options, args):

    label_image = args[0]
    multi_spec_image_path = args[1]

    # read images
    label_1d = read_oneband_image_to_1dArray(label_image)
    multiBand_value_2d = read_multiband_image_to_2dArray(multi_spec_image_path)

    pixel_count = label_1d.shape[0]
    print(label_1d.shape,multiBand_value_2d.shape)

    # print ten pixels for checking
    # for i in range(10):
    #     index = random.randint(1,label_1d.shape[0])
    #     row = index/2384    # 2384 is the width of the input image
    #     col = index%2384
    #     print("row: %d, col: %d, label: %d"%(row,col,label_1d[index]))
    #     print("pixel value: "+ str(multiBand_value_2d[index]))

    # remove the non-ground truth pixels, that is "0" pixel
    back_ground_index = np.where(label_1d==0)
    label_1d = np.delete(label_1d,back_ground_index)
    multiBand_value_2d = np.delete(multiBand_value_2d, back_ground_index,axis=0)

    print("%.2f %% are unclassified (no observation)"%(len(back_ground_index[0])*100.0/pixel_count))
    print('after removing non-ground truth pixels',label_1d.shape, multiBand_value_2d.shape)







if __name__ == "__main__":
    usage = "usage: %prog [options] label_image multi spectral images"
    parser = OptionParser(usage=usage, version="1.0 2018-4-10")
    parser.description = 'Introduction: classification multi-spectral remote sensing images using RNN '

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    # if options.para_file is None:
    #     basic.outputlogMessage('error, parameter file is required')
    #     sys.exit(2)

    main(options, args)