#!/usr/bin/env python
# Filename: split_image
"""
introduction: remove the 0 pixel, as request in the Contest.

First, filter the classified map,only filter on 0 pixel,

Then replace all 0 pixel as class 9 (Non-residential buildings), but class 9 is the most popular pixel 
in ground truth and one best classified map (currently)

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 22 March, 2018
"""

import sys,os,subprocess
from optparse import OptionParser

import numpy as np
from sklearn import metrics
import rasterio

from collections import Counter

def replace_zero_pixel_with_0_other_1(one_band_image):
    """
    create a mask
    :param one_band_image: 
    :return: 
    """

    band, height,width = one_band_image.shape
    one_band_image = one_band_image.reshape(height,width)

    #copy the image value
    new_image = 1*one_band_image

    # copy the image value
    new_image = 1 * one_band_image

    rows, cols = np.where(one_band_image != 0)
    count = len(rows)
    print("Number of non-zero pixel at 1st stage:" + str(count))
    if count != len(cols):
        print(count != len(cols))
        return False

    # filter each pixel
    for i in range(0, count):
        row = rows[i]
        col = cols[i]
        new_image[row, col] = 1

    test = 1

    return new_image


def replace_zero_pixel_with_255(one_band_image):
    """
    replace zeros as 255 for deeplab 
    :param one_band_image: 
    :return: 
    """
    band, height,width = one_band_image.shape
    one_band_image = one_band_image.reshape(height,width)

    #copy the image value
    new_image = 1*one_band_image

    rows,cols = np.where(one_band_image==0)
    count = len(rows)
    print("Number of zero pixel at 1st stage:"+str(count))
    if count!=len(cols):
        print(count!=len(cols))
        return False

    # filter each pixel
    for i in range(0,count):
        row = rows[i]
        col = cols[i]
        new_image[row,col] = 255

    test = 1

    return new_image

def replace_zero_pixel(one_band_image):

    band, height,width = one_band_image.shape
    one_band_image = one_band_image.reshape(height,width)

    #copy the image value
    new_image = 1*one_band_image


    rows,cols = np.where(one_band_image==0)
    count = len(rows)
    print("Number of zero pixel at 1st stage:"+str(count))
    if count!=len(cols):
        print(count!=len(cols))
        return False
    count_non9=0
    count_one_main=0
    count_9=0
    # filter each pixel
    for i in range(0,count):
        row = rows[i]
        col = cols[i]

        # get adjacent value (3*3 window)
        left = max(0,col-1)
        right = min(col+2,width)
        up = max(0,row-1)
        down = min(row+2,height)

        adjacent_values = one_band_image[up:down,left:right]
        adjacent_values = adjacent_values.flatten() # 1d

        if row==16 and col==62:
            test = 1

        #get most_common_class
        class_count = Counter(adjacent_values)
        most_common_class = class_count.most_common(1)
        most_common_class_index =  most_common_class[0][0]

        if most_common_class_index==0:
            # if only have one other non-0 classes, then fill with this class
            # if len(class_count)>=3:
            #     print("hahhahhhhhhhhhhhhhhhhhhhhhh")
            if len(class_count) >= 2:
                second_common_class = class_count.most_common(2)
                second_common_class_index = second_common_class[1][0]
                new_image[row, col] = second_common_class_index
                count_one_main += 1
            else:
                new_image[row, col] = 9
                count_9 += 1

        else:
            new_image[row,col] = most_common_class_index
            count_non9 += 1
            print("filled by most common class, row:%4d, col:%4d" % (row, col))

    print("replace the 0 pixel as 9 (%d), non-9 (%d), and one-main (%d)"%(count_9,count_non9,count_one_main))

    test = 1

    return new_image


def remove_zero_pixel(image_path,output_path):
    if os.path.isfile(image_path) is False:
        print("error, file not exist:" +image_path)
        return False

    clas_map = None
    profile = None
    with rasterio.open(image_path) as img_obj:
        # read the all bands (only have one band)
        indexes = img_obj.indexes
        profile = img_obj.profile
        if len(indexes) != 1:
            print('error, only support one band')
            return False

        clas_map = img_obj.read(indexes)

    if clas_map is None:
        print("error, read map failed")
        return False

    # result = replace_zero_pixel(clas_map)
    result = replace_zero_pixel_with_255(clas_map)

    # result = replace_zero_pixel_with_0_other_1(clas_map)


    if result is False:
        return False

    #save results
    profile.update(dtype=rasterio.uint8,count=1)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(result.astype(rasterio.uint8), 1)

    return True


def main(options, args):
    classified_results = args[0]

    output_path = args[1]

    remove_zero_pixel(classified_results,output_path)





if __name__ == "__main__":
    usage = "usage: %prog [options] classified_result output"
    parser = OptionParser(usage=usage, version="1.0 2018-3-24")
    parser.description = 'Introduction: remove the 0 pixel, as request in the Contest '

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)

