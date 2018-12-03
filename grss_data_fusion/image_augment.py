#!/usr/bin/env python
# Filename: image_augment 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 19 July, 2017
"""


import sys,os,subprocess
from optparse import OptionParser

from imgaug import augmenters as iaa
from skimage import io
# import skimage.transform

import numpy as np

HOME = os.path.expanduser('~')

# path of DeeplabforRS
codes_dir2 = HOME +'/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)
import parameters

import basic_src.basic as basic
import basic_src.io_function as io_function
basic.setlogfile('log_data_augmentation.txt')

# will be update in the main function
num_classes = 0

def remove_unexpected_ids(img_data, img_name):
    '''
    remove unexpected ids after augmentation, it will modify the numpy array
    :param img_data: numpy array of image, should be one band
    :param img_name: file name, help for debug
    :return: True,
    '''

    unique_value = np.unique(img_data)
    if len(unique_value) > num_classes:
        img_data[ img_data > num_classes -1 ] = 0  # unexpected ids are set as zeros (background)
        basic.outputlogMessage('remove unexpected ids (%s) in %s'%(str(unique_value),img_name))

    return True

def Flip(image_np, save_dir, input_filename,is_groud_true):
    """
    Flip image horizontally and vertically
    Args:
        image_np: image_np:  'images' should be either a 4D numpy array of shape (N, height, width, channels)
        save_dir: the directory for saving images
        file_basename: File base name (e.g basename.tif)
        is_groud_true:

    Returns: True if successful, False otherwise

    """
    file_basename = os.path.basename(input_filename)
    basename = os.path.splitext(file_basename)[0]
    ext = os.path.splitext(file_basename)[1]

    flipper = iaa.Fliplr(1.0)  # always horizontally flip each input image; Fliplr(P) Horizontally flips images with probability P.
    images_lr = flipper.augment_image(image_np)  # horizontally flip image 0
    if is_groud_true:
        remove_unexpected_ids(images_lr, input_filename)
    save_path = os.path.join(save_dir,  basename + '_fliplr' + ext)
    io.imsave(save_path, images_lr)
    #
    vflipper = iaa.Flipud(1.0)  # vertically flip each input image with 90% probability
    images_ud = vflipper.augment_image(image_np)  # probably vertically flip image 1
    if is_groud_true:
        remove_unexpected_ids(images_ud, input_filename)
    save_path = os.path.join(save_dir, basename + '_flipud' + ext)
    io.imsave(save_path, images_ud)

    return True

def rotate(image_np, save_dir, input_filename,is_groud_true,degree=[90,180,270]):
    """
    roate image with 90, 180, 270 degree
    Args:
        image_np: image_np:  'images' should be either a 4D numpy array of shape (N, height, width, channels)
        save_dir: the directory for saving images
        input_filename: File base name (e.g basename.tif)
        degree: the degree list for rotation
        is_groud_true:

    Returns: True if successful, False otherwise

    """
    file_basename = os.path.basename(input_filename)
    basename = os.path.splitext(file_basename)[0]
    ext = os.path.splitext(file_basename)[1]

    for angle in degree:
        roate = iaa.Affine(rotate=angle)
        images_r = roate.augment_image(image_np)
        if is_groud_true:
            remove_unexpected_ids(images_r, input_filename)
        save_path = os.path.join(save_dir, basename + '_R'+str(angle) + ext)
        io.imsave(save_path, images_r)

    return True

def scale(image_np, save_dir, input_filename,is_groud_true,scale=[0.5,0.75,1.25,1.5]):
    """
    scale image with 90, 180, 270 degree
    Args:
        image_np: image_np:  'images' should be either a 4D numpy array of shape (N, height, width, channels)
        save_dir: the directory for saving images
        input_filename: File base name (e.g basename.tif)
        scale: the scale list for zoom in or zoom out
        is_groud_true:

    Returns: True is successful, False otherwise

    """
    file_basename = os.path.basename(input_filename)
    basename = os.path.splitext(file_basename)[0]
    ext = os.path.splitext(file_basename)[1]

    for value in scale:
        scale = iaa.Affine(scale=value)
        images_s = scale.augment_image(image_np)
        if is_groud_true:
            remove_unexpected_ids(images_s, input_filename)
        save_path = os.path.join(save_dir, basename + '_S'+str(value).replace('.','') + ext)
        io.imsave(save_path, images_s)

    return True

def blurer(image_np, save_dir, input_filename,is_groud_true,sigma=[1,2]):
    """
    Blur the original images
    Args:
        image_np: image_np:  'images' should be either a 4D numpy array of shape (N, height, width, channels)
        save_dir: the directory for saving images
        input_filename: File base name (e.g basename.tif)
        sigma: sigma value for blurring

    Returns: True if successful, False otherwise

    """

    file_basename = os.path.basename(input_filename)
    basename = os.path.splitext(file_basename)[0]
    ext = os.path.splitext(file_basename)[1]

    for value in sigma:
        save_path = os.path.join(save_dir, basename + '_B' + str(value) + ext)
        if is_groud_true is True:
            # just copy the groud true
            images_b = image_np
        else:
            blurer = iaa.GaussianBlur(value)
            images_b = blurer.augment_image(image_np)
        io.imsave(save_path, images_b)

    return True

def Crop(image_np, save_dir, input_filename,is_groud_true,px = [10,30] ):
    """
    Crop the original images
    Args:
        image_np: image_np:  'images' should be either a 4D numpy array of shape (N, height, width, channels)
        save_dir: the directory for saving images
        input_filename: File base name (e.g basename.tif)
        px:
        is_groud_true

    Returns: True if successful, False otherwise

    """

    file_basename = os.path.basename(input_filename)
    basename = os.path.splitext(file_basename)[0]
    ext = os.path.splitext(file_basename)[1]

    for value in px:
        crop = iaa.Crop(px=value)
        images_s = crop.augment_image(image_np)
        if is_groud_true:
            remove_unexpected_ids(images_s, input_filename)
        save_path = os.path.join(save_dir, basename + '_C'+str(value) + ext)
        io.imsave(save_path, images_s)

    return True

def image_augment(img_path,save_dir,is_groud_true,augment = None):
    if os.path.isfile(img_path) is False:
        print ("Error, File %s not exist"%img_path)
        return False
    if os.path.isdir(save_dir) is False:
        print ("Error, Folder %s not exist"%save_dir)
        return False

    img_test = io.imread(img_path)
    basename = os.path.basename(img_path)

    if 'flip' in  augment:
        if Flip(img_test, save_dir, basename,is_groud_true) is False:
            return False
    if 'rotate' in augment:
        if rotate(img_test, save_dir, basename,is_groud_true, degree=[45, 90, 135]) is False:   #45, 90, 135
            return False
    if 'blur' in augment:
        if blurer(img_test, save_dir, basename,is_groud_true, sigma=[1, 2]) is False:
            return False
    if 'crop' in augment:
        if Crop(img_test, save_dir, basename,is_groud_true, px=[10, 30]) is False:
            return False
    if 'scale' in augment:
        if scale(img_test, save_dir, basename,is_groud_true, scale=[0.75,1.25]) is False:
            return False

    return True



def main(options, args):
    if options.out_dir is None:
        out_dir = "extract_dir"
    else:
        out_dir = options.out_dir

    if os.path.isdir(out_dir) is False:
        os.makedirs(out_dir)

    img_dir = options.img_dir
    extension = options.extension

    is_groud_true = options.ground_truth

    # print(options.para_file)
    augmentation = parameters.get_string_parameters(options.para_file,'data_augmentation')
    augmentation = [item.lower().strip() for item in augmentation.split(',')]
    augmentation = list(filter(None, augmentation))  # remove empty str (in case only have empty string)
    if len(augmentation) < 1  :
        print('No input augmentation requirement (e.g. flip)')
        return True
    # print(augmentation)
    # sys.exit(1)
    # number of classes
    num_classes_noBG = parameters.get_digit_parameters(options.para_file, 'NUM_CLASSES_noBG', None, 'int')
    global num_classes
    num_classes = num_classes_noBG + 1

    #
    ignore_classes = parameters.get_string_parameters(options.para_file,'data_aug_ignore_classes')
    ignore_classes = [item.strip() for item in ignore_classes.split(',')]
    ignore_classes = list(filter(None, ignore_classes))  # remove empty str (in case only have empty string)

    img_list_txt = args[0]
    if os.path.isfile(img_list_txt) is False:
        print ("Error, File %s not exist" % img_list_txt)
        return False
    f_obj = open(img_list_txt)
    index = 1
    files_list = f_obj.readlines()
    for line in files_list:

        # ignore_classes
        if len(ignore_classes)>0:
            found_class = [ line.find(ignore_class) >= 0 for ignore_class in ignore_classes ]
            if True in found_class:
                continue

        file_path  = line.strip()
        file_path = os.path.join(img_dir,file_path+extension)
        print ("Augmentation of image (%d / %d)"%(index,len(files_list)))
        if image_augment(file_path,out_dir,is_groud_true,augment=augmentation) is False:
            print ('Error, Failed in image augmentation')
            return False
        index += 1

    f_obj.close()

    # update img_list_txt
    new_files = io_function.get_file_list_by_ext(extension,'.',bsub_folder=False)
    new_files_noext = [ os.path.splitext(os.path.basename(item))[0]+'\n'  for item in new_files]
    basic.outputlogMessage('save new file list to %s'%img_list_txt)
    with open(img_list_txt,'w') as f_obj:
        f_obj.writelines(new_files_noext)



if __name__ == "__main__":
    usage = "usage: %prog [options] images_txt"
    parser = OptionParser(usage=usage, version="1.0 2017-7-15")
    parser.description = 'Introduction: perform image augmentation '
    # parser.add_option("-W", "--s_width",
    #                   action="store", dest="s_width",
    #                   help="the width of wanted patch")
    # parser.add_option("-H", "--s_height",
    #                   action="store", dest="s_height",
    #                   help="the height of wanted patch")

    parser.add_option("-p", "--para_file",
                      action="store", dest="para_file",
                      help="the parameters file")

    parser.add_option("-g", "--is_ground_truth",
                      action="store_true", dest="ground_truth",default=False,
                      help="indicate whether input image is ground true; should not change the pixel value of groud truth")

    parser.add_option("-d", "--img_dir",
                      action="store", dest="img_dir",
                      help="the folder path for of orginal images")

    parser.add_option("-e", "--extension",
                      action="store", dest="extension",
                      help="the extension of images, for example .png " )

    parser.add_option("-o", "--out_dir",
                      action="store", dest="out_dir",
                      help="the folder path for saving output files")


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    if options.para_file is None:
        print('error, parameter file is required')
        sys.exit(2)

    main(options, args)
