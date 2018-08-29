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

def Flip(image_np, save_dir, input_filename):
    """
    Flip image horizontally and vertically
    Args:
        image_np: image_np:  'images' should be either a 4D numpy array of shape (N, height, width, channels)
        save_dir: the directory for saving images
        file_basename: File base name (e.g basename.tif)

    Returns: True if successful, False otherwise

    """
    file_basename = os.path.basename(input_filename)
    basename = os.path.splitext(file_basename)[0]
    ext = os.path.splitext(file_basename)[1]

    flipper = iaa.Fliplr(1.0)  # always horizontally flip each input image; Fliplr(P) Horizontally flips images with probability P.
    images_lr = flipper.augment_image(image_np)  # horizontally flip image 0
    save_path = os.path.join(save_dir,  basename + '_fliplr' + ext)
    io.imsave(save_path, images_lr)
    #
    vflipper = iaa.Flipud(1.0)  # vertically flip each input image with 90% probability
    images_ud = vflipper.augment_image(image_np)  # probably vertically flip image 1
    save_path = os.path.join(save_dir, basename + '_flipud' + ext)
    io.imsave(save_path, images_ud)

    return True

def rotate(image_np, save_dir, input_filename,degree=[90,180,270]):
    """
    roate image with 90, 180, 270 degree
    Args:
        image_np: image_np:  'images' should be either a 4D numpy array of shape (N, height, width, channels)
        save_dir: the directory for saving images
        input_filename: File base name (e.g basename.tif)
        degree: the degree list for rotation

    Returns: True if successful, False otherwise

    """
    file_basename = os.path.basename(input_filename)
    basename = os.path.splitext(file_basename)[0]
    ext = os.path.splitext(file_basename)[1]

    for angle in degree:
        roate = iaa.Affine(rotate=angle)
        images_r = roate.augment_image(image_np)
        save_path = os.path.join(save_dir, basename + '_R'+str(angle) + ext)
        io.imsave(save_path, images_r)

    return True

def scale(image_np, save_dir, input_filename,scale=[0.5,0.75,1.25,1.5]):
    """
    scale image with 90, 180, 270 degree
    Args:
        image_np: image_np:  'images' should be either a 4D numpy array of shape (N, height, width, channels)
        save_dir: the directory for saving images
        input_filename: File base name (e.g basename.tif)
        scale: the scale list for zoom in or zoom out

    Returns: True is successful, False otherwise

    """
    file_basename = os.path.basename(input_filename)
    basename = os.path.splitext(file_basename)[0]
    ext = os.path.splitext(file_basename)[1]

    for value in scale:
        scale = iaa.Affine(scale=value)
        images_s = scale.augment_image(image_np)
        save_path = os.path.join(save_dir, basename + '_S'+str(value) + ext)
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

def image_augment(img_path,save_dir,is_groud_true):
    if os.path.isfile(img_path) is False:
        print ("Error, File %s not exist"%img_path)
        return False
    if os.path.isdir(save_dir) is False:
        print ("Error, Folder %s not exist"%save_dir)
        return False

    img_test = io.imread(img_path)
    basename = os.path.basename(img_path)

    if Flip(img_test, save_dir, basename) is False:
        return False
    if rotate(img_test, save_dir, basename, degree=[45, 90, 135]) is False:   #45, 90, 135
        return False
    # scale(img_test,save_dir,basename)
    if blurer(img_test, save_dir, basename,is_groud_true, sigma=[1, 2]) is False:
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

    img_list_txt = args[0]
    if os.path.isfile(img_list_txt) is False:
        print ("Error, File %s not exist" % img_list_txt)
        return False
    f_obj = open(img_list_txt)
    index = 1
    files_list = f_obj.readlines()
    for line in files_list:
        file_path  = line.strip()
        file_path = os.path.join(img_dir,file_path+extension)
        print ("Augmentation of image (%d / %d)"%(index,len(files_list)))
        if image_augment(file_path,out_dir,is_groud_true) is False:
            print ('Error, Failed in image augmentation')
            return False
        index += 1

    f_obj.close()



if __name__ == "__main__":
    usage = "usage: %prog [options] images_txt"
    parser = OptionParser(usage=usage, version="1.0 2017-7-15")
    parser.description = 'Introduction: permaform image augmentation '
    # parser.add_option("-W", "--s_width",
    #                   action="store", dest="s_width",
    #                   help="the width of wanted patch")
    # parser.add_option("-H", "--s_height",
    #                   action="store", dest="s_height",
    #                   help="the height of wanted patch")

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

    # if options.para_file is None:
    #     basic.outputlogMessage('error, parameter file is required')
    #     sys.exit(2)

    main(options, args)
