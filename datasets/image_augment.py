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


# will be update in the main function
num_classes = 0

import multiprocessing
from multiprocessing import Pool

code_dir = os.path.join(os.path.dirname(sys.argv[0]), '..')
sys.path.insert(0, code_dir)
import parameters

import basic_src.basic as basic
import basic_src.io_function as io_function

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

def brightness(image_np, save_dir, input_filename,is_groud_true, out_count=1):
    """
    Change the brightness of images: MultiplyAndAddToBrightness
    :param image_np: 'images' should be either a 4D numpy array of shape (N, height, width, channels)
    :param save_dir: the directory for saving images
    :param input_filename: File base name (e.g basename.tif)
    :param is_groud_true: if ground truth, just copy the image
    :return:
    """
    file_basename = os.path.basename(input_filename)
    basename = os.path.splitext(file_basename)[0]
    ext = os.path.splitext(file_basename)[1]

    for idx in range(out_count):
        save_path = os.path.join(save_dir, basename + '_bright' + str(idx) + ext)
        if is_groud_true is True:
            # just copy the groud true
            images_b = image_np
        else:
            brightness = iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30))  # a random value between the range
            images_b = brightness.augment_image(image_np)
        io.imsave(save_path, images_b)

    return True

def contrast(image_np, save_dir, input_filename,is_groud_true, out_count=1):
    """
    Change the constrast of images: GammaContrast
    :param image_np: 'images' should be either a 4D numpy array of shape (N, height, width, channels)
    :param save_dir:
    :param input_filename: File base name (e.g basename.tif)
    :param is_groud_true: if ground truth, just copy the image
    :param :
    :return:
    """
    file_basename = os.path.basename(input_filename)
    basename = os.path.splitext(file_basename)[0]
    ext = os.path.splitext(file_basename)[1]

    for idx in range(out_count):
        save_path = os.path.join(save_dir, basename + '_contrast' + str(idx) + ext)
        if is_groud_true is True:
            # just copy the groud true
            images_con = image_np
        else:
            contrast = iaa.GammaContrast((0.5, 1.5))  # a random gamma value between the range, a large gamma make image darker
            images_con = contrast.augment_image(image_np)
        io.imsave(save_path, images_con)

    return True

def noise(image_np, save_dir, input_filename,is_groud_true, out_count=1):
    """
    Change the constrast of images: AdditiveGaussianNoise
    :param image_np: 'images' should be either a 4D numpy array of shape (N, height, width, channels)
    :param save_dir:
    :param input_filename: File base name (e.g basename.tif)
    :param is_groud_true: if ground truth, just copy the image
    :param :
    :return:
    """
    file_basename = os.path.basename(input_filename)
    basename = os.path.splitext(file_basename)[0]
    ext = os.path.splitext(file_basename)[1]

    for idx in range(out_count):
        save_path = os.path.join(save_dir, basename + '_noise' + str(idx) + ext)
        if is_groud_true is True:
            # just copy the groud true
            images_noise = image_np
        else:
            noise = iaa.AdditiveGaussianNoise(scale=(0, 0.2*255))  # a random gamma value between the range
            images_noise = noise.augment_image(image_np)
        io.imsave(save_path, images_noise)

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
    if 'bright' in augment:
        if brightness(img_test, save_dir, basename,is_groud_true,out_count=2) is False:
            return False
    if 'contrast' in augment:
        if contrast(img_test, save_dir, basename,is_groud_true,out_count=2) is False:
            return False
    if 'noise' in augment:
        if noise(img_test, save_dir, basename,is_groud_true,out_count=2) is False:
            return False

    return True

def augment_one_line(index, line, ignore_classes,file_count,img_dir,extension,out_dir,is_groud_true,augmentation):
    # ignore_classes
    if ignore_classes is not None and len(ignore_classes)>0:
        found_class = [ line.find(ignore_class) >= 0 for ignore_class in ignore_classes ]
        if True in found_class:
            return False

    file_path  = line.strip()
    file_path = os.path.join(img_dir,file_path+extension)
    print ("Augmentation of image (%d / %d)"%(index,file_count))
    if image_augment(file_path,out_dir,is_groud_true,augment=augmentation) is False:
        raise ('Error, Failed in image augmentation')
    return True

def image_augment_main(para_file,img_list_txt,save_list,img_dir,out_dir,extension,is_ground_truth,proc_num):

    basic.setlogfile('log_data_augmentation.txt')

    if os.path.isfile(img_list_txt) is False:
        raise IOError("File %s not exist" % img_list_txt)

    if os.path.isdir(out_dir) is False:
        os.makedirs(out_dir)

    if img_dir != out_dir:
        raise ValueError('set image dir and output dir be the same, making it easy to update image list')

    # print(options.para_file)
    augmentation = parameters.get_string_list_parameters_None_if_absence(para_file,'data_augmentation')
    if augmentation is None or len(augmentation) < 1  :
        basic.outputlogMessage('No input augmentation requirement (e.g. flip), skip data augmentation')
        return True

    # number of classes
    num_classes_noBG = parameters.get_digit_parameters_None_if_absence(para_file, 'NUM_CLASSES_noBG', 'int')
    global num_classes
    num_classes = num_classes_noBG + 1

    # ignored classes
    ignore_classes = parameters.get_string_list_parameters_None_if_absence(para_file,'data_aug_ignore_classes')


    with open(img_list_txt, 'r') as f_obj:
        files_list = f_obj.readlines()
    file_count = len(files_list)
    index = 1
    # for line in files_list:
    #
    #     # ignore_classes
    #     if ignore_classes is not None and len(ignore_classes)>0:
    #         found_class = [ line.find(ignore_class) >= 0 for ignore_class in ignore_classes ]
    #         if True in found_class:
    #             continue
    #
    #     file_path  = line.strip()
    #     file_path = os.path.join(img_dir,file_path+extension)
    #     print ("Augmentation of image (%d / %d)"%(index,file_count))
    #     if image_augment(file_path,out_dir,is_ground_truth,augment=augmentation) is False:
    #         print ('Error, Failed in image augmentation')
    #         return False
    #     index += 1

    parameters_list = [
        (index + 1, line, ignore_classes, file_count, img_dir, extension, out_dir, is_ground_truth, augmentation)
        for index, line in enumerate(files_list)]
    theadPool = Pool(proc_num)  # multi processes
    results = theadPool.starmap(augment_one_line, parameters_list)  # need python3
    augmented = [1 for item in results if item is True]
    # print(sum(augmented))

    if sum(augmented) < file_count:
        basic.outputlogMessage('Some of the images belong to %s are ignored' % ','.join(ignore_classes))

    # update img_list_txt (img_dir is the same as out_dir)
    new_files = io_function.get_file_list_by_ext(extension, out_dir, bsub_folder=False)
    new_files_noext = [os.path.splitext(os.path.basename(item))[0] + '\n' for item in new_files]
    basic.outputlogMessage('save new file list to %s' % save_list)
    with open(save_list, 'w') as f_obj:
        f_obj.writelines(new_files_noext)


def main(options, args):

    if options.out_dir is None:
        out_dir = "data_augmentation_dir"
    else:
        out_dir = options.out_dir

    # in most case, img_dir is the same as out_dir
    img_dir = options.img_dir

    extension = options.extension

    is_ground_truth = options.ground_truth
    proc_num = options.process_num
    para_file = options.para_file
    save_list = options.save_list

    img_list_txt = args[0]
    image_augment_main(para_file, img_list_txt,save_list, img_dir, out_dir, extension, is_ground_truth, proc_num)





if __name__ == "__main__":
    usage = "usage: %prog [options] images_txt"
    parser = OptionParser(usage=usage, version="1.0 2017-7-15")
    parser.description = 'Introduction: perform image augmentation '


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

    parser.add_option("-l", "--save_list",default='images_including_aug.txt',
                      action="store", dest="save_list",
                      help="the text file for saving the images after data augmentation")
    parser.add_option("-n", "--process_num", type=int,
                      action="store", dest="process_num", default=4,
                      help="the process number for parallel computing")


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)


    if options.para_file is None:
        print('error, parameter file is required')
        sys.exit(2)


    main(options, args)
