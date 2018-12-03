#!/usr/bin/env python
# Filename: calculate_meanvalue 
"""
introduction: calculate the mean value of each band for the all input images

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 17 April, 2017

modify on Dec 3, 2018
"""

import os,sys
from optparse import OptionParser


# landuse_path = HOME + '/codes/PycharmProjects/Landuse_DL'
# sys.path.append(landuse_path+'/datasets')

HOME = os.path.expanduser('~')
# path of DeeplabforRS
codes_dir2 = HOME +'/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)

from basic_src import io_function
from basic_src import basic
from basic_src import RSImage
from basic_src.RSImage import RSImageclass


def get_first_path_in_line(input_path):
    # if multi files in one line, then only consider the first file
    image_path = input_path
    if os.path.isfile(image_path) is False:
        image_path = image_path.split()[0]  # only consider first image
        if os.path.isfile(image_path) is False:
            basic.outputlogMessage('File not exist %s ' % image_path)
            return False
    return image_path


def cal_the_mean_of_bands(input_path):

    # if multi files in one line, then only consider the first file
    image_path = get_first_path_in_line(input_path)
    if image_path is False:
        return False

    img_obj = RSImageclass()
    if img_obj.open(image_path):
        width = img_obj.GetWidth()
        height = img_obj.GetHeight()
        mean_of_bands = RSImage.get_image_mean_value(image_path)

        if mean_of_bands is False:
            return (False,False,False)
        return (width,height,mean_of_bands)
    else:
        basic.outputlogMessage('error, Open image %s failed' %image_path)
        return (False,False,False)


def calculate_mean_of_images(images_list):
    if len(images_list)<1:
        basic.outputlogMessage('No image in the list')
        return False
    # get band number
    img_obj = RSImageclass()
    first_img = get_first_path_in_line(images_list[0])
    if first_img is False:
        return False
    img_obj.open(first_img)
    band_count = img_obj.GetBandCount()
    img_obj = None

    mean_of_images = []  # each band has one value
    for i in range(0,band_count):
        mean_of_images.append(0.0)
    total_pixel = 0

    number = 0
    for image in images_list:
        (width,height,mean_of_band) = cal_the_mean_of_bands(image)
        pixel_count = width*height
        number = number +1
        print ('%d / %d'%(number,len(images_list)))
        if width is False:
            return False
        if len(mean_of_band) != band_count:
            basic.outputlogMessage('error, band count is different')
            return False
        for i in range(0, band_count):
            mean_of_images[i] = (mean_of_images[i]*total_pixel + mean_of_band[i]*pixel_count)

        total_pixel = total_pixel + width*height
        for i in range(0, band_count):
            mean_of_images[i] = mean_of_images[i]/total_pixel


    for i in range(0, band_count):
        basic.outputlogMessage('band {}: mean {}'.format(i+1, mean_of_images[i]))

    f_obj =  open('mean_value.txt','w')
    f_obj.writelines('total image count {} \n'.format(len(images_list)))
    for i in range(0,len(mean_of_images)):
        f_obj.writelines('band {} \n'.format(i + 1))
    for i in range(0,len(mean_of_images)):
        f_obj.writelines('mean_value: {} \n'.format(mean_of_images[i]))
    f_obj.close()

    return mean_of_images


def main(options, args):

    input_path = args[0]
    if os.path.isdir(input_path) is True:
        if options.ext is None:
            basic.outputlogMessage('image file extenstion is need, type help for more information')
            return False

        file_list = io_function.get_file_list_by_ext(options.ext, input_path, bsub_folder=True)
        f_obj = open('images_list.txt','w')
        f_obj.writelines(["%s\n" % item for item in file_list])
        f_obj.close()
    elif os.path.isfile(input_path):
        f_obj = open(input_path,'r')
        file_list = f_obj.readlines()
        f_obj.close()
    else:
        basic.outputlogMessage('input error:  %s'%input_path)
        return False

    for i in range(0,len(file_list)):
        file_list[i] = file_list[i].strip()

    return calculate_mean_of_images(file_list)


if __name__=='__main__':
    usage = "usage: %prog [options] file_list or image_folder"
    parser = OptionParser(usage=usage, version="1.0 2017-4-17")

    parser.add_option("-e", "--image_ext", action="store", dest="ext",
                      help="the extension of the image files, like .tif (don't miss the dot), \
                      need this when the input is a folder")

    (options, args) = parser.parse_args()
    if len(args)<1:
        parser.print_help()
        exit(0)

    main(options, args)