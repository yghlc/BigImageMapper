#!/usr/bin/env python
# Filename: build_RS_data 
"""
introduction: Convert Remote Sensing images (one band or three bands) to TF record for training

modified from build_voc2012_data.py

add time: 18 March, 2018
"""

import glob
import math
import os.path
import sys
import build_data
import tensorflow as tf

HOME = os.path.expanduser('~')
basicCodes_path = HOME + '/codes/PycharmProjects/DeeplabforRS'
sys.path.append(basicCodes_path)

# package for remote sensing images
import rasterio
from basic_src.RSImage import RSImageclass
import basic_src.basic as  basic
import split_image
import rasterio
import numpy as np
import parameters

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_folder',
                           './Lidar_GeoTiff_Rasters/Intensity_C1',
                           'Folder remote sensing images.')

tf.app.flags.DEFINE_string(
    'label_image_folder',
    './GT',
    'Folder containing label images.')

tf.app.flags.DEFINE_string(
    'list_file',
    './image_list',
    'file containing lists for remote sensing and label images')

tf.app.flags.DEFINE_bool('is_training', True,
                         'whether these data are for training (including labels)? ')

tf.app.flags.DEFINE_string(
    'para_file',
    'para.ini',
    'The parameter file containing file path and parameters')

tf.app.flags.DEFINE_string(
    'output_dir',
    './tfrecord',
    'Path to save converted SSTable of TensorFlow examples.')




_NUM_SHARDS = 4


class patchclass(object):
    """
    store the information of each patch (a small subset of the remote sensing images)
    """
    def __init__(self,org_img,boundary):
        self.org_img = org_img  # the original remote sensing images of this patch
        self.boundary=boundary      # the boundary of patch (xoff,yoff ,xsize, ysize) in pixel coordinate
    def boundary(self):
        return self.boundary


def read_patch(patch_obj):
    """
    Read all the pixel of the patch
    :param patch_obj: the instance of patchclass
    :return: The array of pixel value
    """
    # window structure; expecting ((row_start, row_stop), (col_start, col_stop))
    boundary = patch_obj.boundary #(xoff,yoff ,xsize, ysize)
    window = ((boundary[1],boundary[1]+boundary[3])  ,  (boundary[0],boundary[0]+boundary[2]))
    with rasterio.open(patch_obj.org_img) as img_obj:
        # read the all bands
        indexes = img_obj.indexes
        data = img_obj.read(indexes,window=window)

        # replace the nodata as zeros (means background)
        if img_obj.profile.has_key('nodata'):
            nodata = img_obj.profile['nodata']
            data[np.where(data==nodata)] = 0

        return data

def save_patch_oneband_8bit(patch_obj,img_data,save_path):
    """
    save a patch of image, currently support oneband with 8bit
    :param patch_obj: the patch object, contain the boundary and original (reference) image path
    :param img_data: numpy array of a new image, could be one band or multi band
    :param save_path: Save file path
    :return: True if sucessufully, False otherwise
    """

    # reference image path
    org_img_path = patch_obj.org_img
    dataset_name = os.path.splitext(os.path.basename(org_img_path))[0]

    # save the segmentation map
    boundary = patch_obj.boundary
    # boundary = [item[0] for item in boundary]
    xsize = boundary[2]
    ysize = boundary[3]

    # check width and height
    height,width = img_data.shape
    if xsize != width or ysize != height:
        basic.outputlogMessage("Error, the Size of the saved numpy array is different from the original patch")
        return False

    window = ((boundary[1], boundary[1] + ysize), (boundary[0], boundary[0] + xsize))

    # img.save('inf_result/'+fName+'.png')
    with rasterio.open(org_img_path) as org:
        profile = org.profile
        new_transform = org.window_transform(window)
    # calculate new transform and update profile (transform, width, height)

    profile.update(dtype=rasterio.uint8, count=1, transform=new_transform, width=xsize, height=ysize)
    # set the block size, it should be a multiple of 16 (TileLength must be a multiple of 16)
    if profile.has_key('blockxsize') and profile['blockxsize'] > xsize:
        if xsize % 16 == 0:
            profile.update(blockxsize=xsize)
        else:
            profile.update(blockxsize=16)  # profile.update(blockxsize=16)
    if profile.has_key('blockysize') and profile['blockysize'] > ysize:
        if ysize % 16 == 0:
            profile.update(blockysize=ysize)
        else:
            profile.update(blockysize=16)  # profile.update(blockxsize=16)


    with rasterio.open(save_path, "w", **profile) as dst:
        dst.write(img_data, 1)

def split_patches_into_batches(patches, batch_size):
    """
    split the patches of an image to many small group (batch).
    In each group, the patches should have the same width and height
    :param patches: a 1-d list of patches
    :param batch_size: batch_size, e.g., 4, or 5 depends on the GPU memory
    :return: a 2D list, contains all the groups
    """
    # check input value
    assert len(patches) > 0
    assert batch_size > 0

    batches = []
    print('splitting patches to small batches (groups), wait a moment')

    # ensure the patch is still in the original sequence.
    # each batch, has the patches with size width and height (or band?)
    # therefore, each the size of each batch <= batch_size, but cannot know its value

    # group the patches based on the requirement above
    patch_count = len(patches)
    currentIdx = 0

    while(currentIdx < patch_count):

        a_batch = [patches[currentIdx]] # get the first patch
        boundary_1st = patches[currentIdx].boundary
        # width and height of the first patch
        width = boundary_1st[2]
        height = boundary_1st[3]

        currentIdx += 1

        while(len(a_batch) < batch_size and currentIdx < patch_count):
            # get the next patch
            patch_obj = patches[currentIdx]

            # check its width and height
            boundary = patch_obj.boundary
            xsize = boundary[2]  # width
            ysize = boundary[3]  # height
            if xsize!=width or ysize!=height:
                break
            else:
                a_batch.append(patch_obj)
                currentIdx += 1

        batches.append(a_batch)

    return batches


def check_input_image_and_label(image_path, label_path):
    """
    check the input image and label, they should have same width, height, and projection
    :param image_path: the path of image
    :param label_path: the path of label
    :return: (width, height) of image if successful, Otherwise (None, None).
    """

    img_obj = RSImageclass()
    if img_obj.open(image_path) is False:
        assert False
    label_obj = RSImageclass()
    if label_obj.open(label_path) is False:
        assert False
    # check width and height
    width = img_obj.GetWidth()
    height = img_obj.GetHeight()
    if width != label_obj.GetWidth() or height != label_obj.GetHeight():
        basic.outputlogMessage("Error, not the same width and height of image (%s) and label (%s)"%(image_path,label_path))
        assert False

    # check resolution
    if img_obj.GetXresolution() != label_obj.GetXresolution() or img_obj.GetYresolution() != label_obj.GetYresolution():
        basic.outputlogMessage(
            "Error, not the same resolution of image (%s) and label (%s)" % (image_path, label_path))
        assert False

    # check projection
    if img_obj.GetProjection() != label_obj.GetProjection():
        basic.outputlogMessage(
            "warning, not the same projection of image (%s) and label (%s)" % (image_path, label_path))
    #     assert False

    return (width, height)

def make_dataset(root,list_txt,patch_w,patch_h,adj_overlay_x,adj_overlay_y,train=True):
    """
    get the patches information of the remote sensing images in one list file.
    :param root: data root
    :param list_txt: a list file contain images (one row contain one train image and one label
    image with space in the center if the input is for training; one row contain one image if it is for inference)
    :param patch_w: the width of the expected patch
    :param patch_h: the height of the expected patch
    :param adj_overlay: the extended distance (in pixel) to adjacent patch, make each patch has overlay with adjacent patch
    :param train:  indicate training or inference
    :return: dataset (2D list), 1st dimension is for images, 2nd dimension is for all patches in the corresponding image
    """
    dataset = []

    if os.path.isfile(list_txt) is False:
        basic.outputlogMessage("error, file %s not exist"%list_txt)
        assert False

    with open(list_txt) as file_obj:
        files_list = file_obj.readlines()
    if len(files_list) < 1:
        basic.outputlogMessage("error, no file name in the %s" % list_txt)
        assert False

    if train:
        for line in files_list:
            names_list = line.split()
            if len(names_list) < 1: # empty line
                continue
            image_name = names_list[0]
            label_name = names_list[1].strip()

            img_path = os.path.join(root,image_name)
            label_path = os.path.join(root,label_name)
            #
            (width,height) = check_input_image_and_label(img_path,label_path)

            # split the image and label
            patch_boundary = split_image.sliding_window(width, height, patch_w, patch_h, adj_overlay_x,adj_overlay_y)

            patches_of_a_image = []
            for patch in patch_boundary:
                # need to handle the patch with smaller size, also deeplab can handle this
                # remove the patch small than model input size
                # if patch[2] < crop_width or patch[3] < crop_height:   # xSize < 480 or ySize < 480
                #     continue
                img_patch = patchclass(img_path,patch)
                label_patch = patchclass(label_path,patch)
                patches_of_a_image.append([img_patch, label_patch])

            dataset.append(patches_of_a_image)

    else:
        for line in files_list:
            names_list = line.split()
            image_name = names_list[0].strip()

            img_path = os.path.join(root,image_name)
            #
            img_obj = RSImageclass()
            if img_obj.open(img_path) is False:
                assert False
            width = img_obj.GetWidth()
            height = img_obj.GetHeight()

            # split the image and label
            patch_boundary = split_image.sliding_window(width, height, patch_w, patch_h, adj_overlay_x,adj_overlay_y)

            patches_of_a_image = []
            for patch in patch_boundary:
                # need to handle the patch with smaller size
                # if patch[2] < crop_width or patch[3] < crop_height:   # xSize < 480 or ySize < 480
                #     continue
                img_patch = patchclass(img_path,patch)

                patches_of_a_image.append(img_patch)

            dataset.append(patches_of_a_image)

    return dataset


def _convert_dataset(dataset_patches,train=True):
    """Converts the specified dataset split to TFRecord format.

    Args:
      dataset_split: The patches of remote sensing dataset (e.g., train, test).
      train: True means that one patch contains both a remote sensing image and a label image,
             False means that one patch only contains a Remote sensing image

    Raises:
      RuntimeError: If loaded image and label have different shape.
    """
    # notes: http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
    # notes: https://gist.github.com/swyoon/8185b3dcf08ec728fb22b99016dd533f

    num_images = len(dataset_patches)
    num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

    # use the first image name as dataset_name
    org_img = dataset_patches[0][0].org_img
    dataset_name = os.path.splitext(os.path.basename(org_img))[0]

    if train:

        for idx in range(num_images):

            img_patch, gt_patch = dataset_patches[idx]

            image_data = read_patch(img_patch)
            label_data = read_patch(gt_patch) #build_data.ImageReader('png', channels=1)

            for shard_id in range(_NUM_SHARDS):
                output_filename = os.path.join(
                    FLAGS.output_dir,
                    '%s-%05d-of-%05d.tfrecord' % (dataset_name, shard_id, _NUM_SHARDS))
                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_idx = shard_id * num_per_shard
                    end_idx = min((shard_id + 1) * num_per_shard, num_images)
                    for i in range(start_idx, end_idx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                            i + 1, num_images, shard_id))
                        sys.stdout.flush()
                        # Read the image.
                        org_img = img_patch.org_img
                        file_name = os.path.splitext(os.path.basename(org_img))[0] + '_' + str(i)

                        # image_filename = os.path.join(
                        #     FLAGS.image_folder, filenames[i] + '.' + FLAGS.image_format)
                        # image_data = tf.gfile.FastGFile(image_filename, 'r').read()
                        # height, width = image_reader.read_image_dims(image_data)
                        image_shape = image_data.shape
                        height, width = image_shape[1],image_shape[2]
                        # # Read the semantic segmentation annotation.
                        # seg_filename = os.path.join(
                        #     FLAGS.semantic_segmentation_folder,
                        #     filenames[i] + '.' + FLAGS.label_format)
                        # seg_data = tf.gfile.FastGFile(seg_filename, 'r').read()
                        label_shape = label_data.shape
                        label_height, label_width = image_shape[1],image_shape[2]
                        # seg_height, seg_width = label_reader.read_image_dims(seg_data)
                        if height != label_height or width != label_width:
                            raise RuntimeError('Shape mismatched between image and label.')
                        # Convert to tf example.
                        example = build_data.image_seg_to_tfexample(
                            image_data, file_name, height, width, label_data)
                        tfrecord_writer.write(example.SerializeToString())
                sys.stdout.write('\n')
                sys.stdout.flush()

    else:
        # img_patch = self.patches[idx]
        # patch_info = [img_patch.org_img, img_patch.boundary]
        # # img_name_noext = os.path.splitext(os.path.basename(img_patch.org_img))[0]+'_'+str(idx)
        # img = read_patch(img_patch)
        # # img.resize(self.nRow,self.nCol)
        # img = img[:, 0:self.nRow, 0:self.nCol]
        # img = np.atleast_3d(img).astype(np.float32)
        # if (img.max() - img.min()) < 0.01:
        #     pass
        # else:
        #     img = (img - img.min()) / (img.max() - img.min())
        # img = torch.from_numpy(img).float()
        # return img, patch_info

        # dataset = os.path.basename(dataset_split)[:-4]
        # sys.stdout.write('Processing ' + dataset)



        pass



def main(unused_argv):
    # dataset_splits = glob.glob(os.path.join(FLAGS.list_file, '*.txt'))
    #
    # for dataset_split in dataset_splits:
    #     _convert_dataset(dataset_split)

    #how about the mean value?

    #split images
    data_root = FLAGS.image_folder
    list_txt = FLAGS.list_file

    ############## dataset processing
    parameters.set_saved_parafile_path(FLAGS.para_file)
    patch_w = parameters.get_digit_parameters("", "train_patch_width", None, 'int')
    patch_h = parameters.get_digit_parameters("", "train_patch_height", None, 'int')
    overlay_x = parameters.get_digit_parameters("", "train_pixel_overlay_x", None, 'int')
    overlay_y = parameters.get_digit_parameters("", "train_pixel_overlay_y", None, 'int')

    patches = make_dataset(data_root, list_txt, patch_w, patch_h, overlay_x,overlay_y, train=FLAGS.is_training)

    os.system("mkdir -p " + FLAGS.output_dir)

    #convert images
    patches_1d = [item for alist in patches for item in alist]  # convert 2D list to 1D
    _convert_dataset(patches_1d,train=FLAGS.is_training)
    pass



if __name__ == '__main__':
  tf.app.run()

