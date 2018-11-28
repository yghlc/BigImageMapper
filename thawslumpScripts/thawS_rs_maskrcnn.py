"""
Mask R-CNN
Configurations and data loading code for Planet cubSat images (mapping thaw slumps)

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla


Modify by: Lingcao Huang, 10-Nov-2018

Run this codes using Python > 3.4

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 thawS_rs_maskrcnn.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 thawS_rs_maskrcnn.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 thawS_rs_maskrcnn.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 thawS_rs_maskrcnn.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 thawS_rs_maskrcnn.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)

import random

# import skimage
import skimage.io
import skimage.color

import cv2
import subprocess

# check python version
from distutils.version import LooseVersion
print(sys.version)
if (LooseVersion(sys.version) > LooseVersion('3.4')) is False:
    raise EnvironmentError('Require Python version > 3.4')

NO_DATA = 255
para_file = 'para_mrcnn.ini'
inf_list_file = 'inf_image_list.txt'
inf_output_dir = 'inf_results'

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# from pycocotools import mask as maskUtils

# Root directory of the project
HOME = os.path.expanduser('~')
codes_dir = HOME +'/codes/PycharmProjects/object_detection/yghlc_Mask_RCNN'

# current folder, usually is where I run the codes
curr_dir = os.getcwd()

# Import Mask RCNN
sys.path.append(codes_dir)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize
from mrcnn.model import log

# path of DeeplabforRS
codes_dir2 = HOME +'/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)

landuse_path = HOME + '/codes/PycharmProjects/Landuse_DL'
sys.path.append(landuse_path+'/datasets')
import build_RS_data as build_RS_data

import parameters
from basic_src import io_function


############################################################
#  Configurations
############################################################


class PlanetConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "planet"

    # Backbone network architecture
    BACKBONE = "resnet101"

    # We use a GPU with 12GB memory, which can fit two images for 1024*2014.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2 #??

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # two class. i.e. thaw slumps and "non-thaw-slumps but similar"

    #add more
    # the large side, and that determines the image shape, the size of the split images < 480 by 480
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small on Planet image
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels, it seem  only accept five values

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


############################################################
#  Dataset
############################################################

class PlanetDataset(utils.Dataset):
    def load_Planet(self, image_dir,label_dir,subset):
        """Load a subset of the Planet images
        image_dir: The root directory of the Planet images (already split ).
        subset: What to load (train, val, minival, valminusminival)
        class_ids: If provided, only loads images that have the given classes.
        """
        # Add classes
        self.add_class("planet_TS", 1, "thawslump")    # source, class_id, class_name
        self.add_class("planet_noTS", 2, "thawslump_similar")  # source, class_id, class_name

        # Add images in the image_dir
        if subset=='train':
            images_list = 'list/train_list.txt'
        elif subset == 'val':
            images_list = 'list/val_list.txt'
        else:
            raise ValueError("'{}' is not recognized. Use 'train' or 'val' ".format(subset))

        # dataset = os.path.basename(dataset_split)[:-4]
        filenames = [x.strip('\n') for x in open(images_list, 'r')]

        for i, image_name in enumerate(filenames):
            # source, image_id, path, **kwargs
            image_path = os.path.join(os.path.abspath(image_dir),image_name+'.png')
            label_path = os.path.join(os.path.abspath(label_dir),image_name+'.png')
            img_source = 'unknow'
            if image_path.find('class_1')>0:
                img_source = 'planet_TS'
            elif image_path.find('class_2')>0:
                img_source = 'planet_noTS'
            else:
                raise ValueError('unknow class in file name: %s '%image_name)
            self.add_image(img_source, image_id=i, path=image_path,
                           label_path = label_path, patch=image_name)


    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        The mask is orginal for semantic segmentation using DeepLab, now convert it. for instance segmentation

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        #
        image_info = self.image_info[image_id]

        instance_masks = []
        class_ids = []

        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        label_path = image_info['label_path']
        # print(label_path)

        # Load image
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED) #skimage.io.imread(label_path)
        # The label is a one band, 8 bit image
        if label.ndim != 2: # one band images only have a shape of (height, width), instead of (height, width, nband)
            raise ValueError('The band count of label must be 1')
        if label.dtype != np.uint8:
            raise ValueError('The label image should have data type of uint8')

        # set nodata pixels to background, that is 0
        label [label == NO_DATA ] = 0

        height, width = label.shape
        unique_ids, counts = np.unique(label, return_counts=True)
        if len(unique_ids) > PlanetConfig.NUM_CLASSES:
            raise ValueError(str(unique_ids)+' its count is: %d but number of classes is: %d'
                             %(len(unique_ids),PlanetConfig.NUM_CLASSES))
        if max(unique_ids) >=  PlanetConfig.NUM_CLASSES:
            raise ValueError(str(unique_ids) + ' the maximum of id is greater than () the number of classes is: %d'
                             % (PlanetConfig.NUM_CLASSES))

        # # create the mask for each class (excluding the background)
        # for id, count in zip(unique_ids,counts):
        #     # ignore background
        #     if id==0:
        #         continue
        #     # Some objects are so small that they're less than 1 pixel area
        #     # and end up rounded out. Skip those objects.
        #     if count < 1:
        #         continue
        #     m = np.zeros([height, width], dtype=bool)
        #
        #     m[label == id] = True
        #
        #     instance_masks.append(m)
        #     class_ids.append(id)

        # image_data = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        # height, width = image_data.shape

        # test_idx = image_data != 0
        # image_data[image_data != 0] = 255
        # cv2.imwrite('mask_255.tif', image_data)

        # if no objects on this images
        if max(unique_ids) == 0:
            # Call super class to return an empty mask
            return super(PlanetDataset, self).load_mask(image_id)
        else:

            image, contours, hierarchy = cv2.findContours(label, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for idx in range(0, len(contours)):
                # create seed mask
                seed_masks = np.zeros((height, width), np.int32)
                # print('idx:', idx)

                point = contours[idx][0][0]  # [col,row]
                # print('point:', point)
                id = label[point[1], point[0]] # [row,col]
                # print('class_id:', id)
                if id not in unique_ids or id==0:
                    raise ValueError('class_id: %d not in the label images or is zeros (Backgroud)'%id)
                cv2.drawContours(seed_masks, contours, idx, (idx + 1), -1)  # -1 for filling inside

                seed_masks = seed_masks.astype(np.uint8)
                instance_masks.append(seed_masks)
                class_ids.append(id)

                # test
                # seed_masks = seed_masks * 50
                # cv2.imwrite('seed_masks_255_%d_inst_%d.tif'%(image_id,idx), seed_masks * 50)


        # Pack instance masks into an array, if there are objects
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(PlanetDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "planet":
            return info["patch"]
        else:
            super(self.__class__).image_reference(image_id)

############################################################
#  COCO Evaluation
############################################################

def build_planet_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Convert the results to the format of semantic segmentation format
    """
    pass


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """

def muti_inf_remoteSensing_image(model,image_path=None):
    '''
    use multiple scale (different patch size) for inference, then merge them using non_max_suppression
    :param model: trained model
    :param image_path:
    :return: True if successful, False otherwise
    '''

    # get parameters
    inf_image_dir = parameters.get_string_parameters(para_file, 'inf_images_dir')
    muti_patch_w = parameters.get_string_parameters(para_file, "muti_inf_patch_width")
    muti_patch_h = parameters.get_string_parameters(para_file, "muti_inf_patch_height")
    muti_overlay_x = parameters.get_string_parameters(para_file, "muti_inf_pixel_overlay_x")
    muti_overlay_y = parameters.get_string_parameters(para_file, "muti_inf_pixel_overlay_y")
    final_keep_classes = parameters.get_string_parameters(para_file, "final_keep_classes")

    nms_iou_threshold = parameters.get_digit_parameters(para_file, "nms_iou_threshold", None, 'float')

    patch_w_list = [int(item) for item in muti_patch_w.split(',')]
    patch_h_list = [int(item) for item in muti_patch_h.split(',')]
    overlay_x_list = [int(item) for item in muti_overlay_x.split(',')]
    overlay_y_list = [int(item) for item in muti_overlay_y.split(',')]
    if final_keep_classes == '':
        final_keep_classes = None
    else:
        final_keep_classes = [int(item) for item in final_keep_classes.split(',')]

    # inference and save to json files
    for patch_w,patch_h,overlay_x,overlay_y in zip(patch_w_list,patch_h_list,overlay_x_list,overlay_y_list):
        inf_rs_image_json(model, patch_w, patch_h, overlay_x, overlay_y, inf_image_dir)

    # load all boxes of images
    with open(inf_list_file) as file_obj:
        files_list = file_obj.readlines()
        for img_idx, image_name in enumerate(files_list):
            file_pattern = os.path.join(inf_output_dir, 'I%d_patches_*_*_*'%img_idx) # e.g., I0_patches_320_320_80_80
            proc = subprocess.Popen('ls -d ' + file_pattern, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            profiles, err = proc.communicate()
            json_folder_list = profiles.split()
            if len(json_folder_list) < 1:
                raise IOError('No folder containing json files in %s'%inf_output_dir)

            # bytes to str
            if isinstance(json_folder_list[0],bytes):
                json_folder_list = [item.decode() for item in json_folder_list]

            print('loading json files of image :%d,  in %s'%(img_idx,','.join(json_folder_list)))
            # load all the boxes and scores
            mrcnn_r_list = []   # the results dict from rcnn, contains all the information
            mask_files = []     # mask file for instance, each box has a mask file
            boxes = []          # boxes of instance
            class_ids = []      # class id of each box
            patch_indices = []  # index of patch, on which contains the box
            scores = []         # scores of boxes
            json_files_list = [] # json files of patches
            for json_folder in json_folder_list:
                file_list = io_function.get_file_list_by_ext('.txt',json_folder,bsub_folder=False)
                json_files_list.extend(file_list)

            # load and convert coordinates, don't load mask images in this stage
            patch_idx = 0
            for json_file in json_files_list:
                mrcnn_r = build_RS_data.load_instances_patch(json_file, bNMS=True,bReadMaks=False,final_classes=final_keep_classes)
                mrcnn_r_list.append(mrcnn_r)  # this corresponds to json_files_list
                if mrcnn_r is not None:     # this will ignore the patches without instances, it is fine hlc 2018-11-25
                    mask_files.extend(mrcnn_r['masks'])
                    boxes.extend(mrcnn_r['rois'])
                    scores.extend(mrcnn_r['scores'])
                    class_ids.extend(mrcnn_r['class_ids'])
                    patch_indices.extend([patch_idx]*len(mrcnn_r['rois']))
                patch_idx += 1

            # Apply non-max suppression
            keep_idxs = utils.non_max_suppression(np.array(boxes), np.array(scores), nms_iou_threshold)
            # boxes_keep = [r for i, r in enumerate(boxes) if i in keep_ixs]

            # convert kept patches to label images
            for idx,keep_idx in enumerate(keep_idxs):

                patch_idx = patch_indices[keep_idx]         # the index in original patches

                # load mask (in the previous step, we did not load masks)
                mask_file = mask_files[keep_idx]
                patch_dir = os.path.dirname(json_files_list[patch_idx])
                org_img_name = mrcnn_r_list[patch_idx]['org_img']
                b_dict = mrcnn_r_list[patch_idx]['patch_boundary']
                patch_boundary = (b_dict['xoff'],b_dict['yoff'],b_dict['xsize'],b_dict['ysize'])
                img_patch = build_RS_data.patchclass(os.path.join(inf_image_dir,org_img_name),patch_boundary)

                # the mask only contains one instances
                # masks can overlap each others, but instances should not overlap each other
                # non-instance pixels are zeros,which will be set as non-data when performing gdal_merge.pyg
                mask = cv2.imread(os.path.join(patch_dir, mask_file), cv2.IMREAD_UNCHANGED)
                mask [mask == 255] = class_ids[keep_idx]  # when save mask,  mask*255 for display


                print('Save mask of instances:%d on Image:%d , shape:(%d,%d)' %
                      (idx,img_idx, mask.shape[0], mask.shape[1]))

                # short the file name to avoid  error of " Argument list too long", hlc 2018-Oct-29
                file_name = "I%d_%d" % (img_idx, idx)

                save_path = os.path.join(inf_output_dir, file_name + '.tif')
                if build_RS_data.save_patch_oneband_8bit(img_patch, mask.astype(np.uint8), save_path) is False:
                    return False



    return True

def inf_rs_image_json(model,patch_w,patch_h,overlay_x,overlay_y,inf_image_dir):
    """
    inference on images and save to json files
    :param model: trained model
    :param patch_w: split width
    :param patch_h: split height
    :param overlay_x: overlay pixels in x direction
    :param overlay_y: overlay pixels in y direction
    :param inf_image_dir: the folder contains images for inference
    :return: True if no error, otherwise, False
    """

    # global inf_list_file
    # if image_path is not None:
    #     with open('inf_image_list.txt','w') as f_obj:
    #         f_obj.writelines(image_path)
    #         inf_list_file = 'inf_image_list.txt'

    data_patches_2d = build_RS_data.make_dataset(inf_image_dir,inf_list_file,
                patch_w,patch_h,overlay_x,overlay_y,train=False)

    if len(data_patches_2d)< 1:
        return False

    total_patch_count = 0
    for img_idx, aImage_patches in enumerate(data_patches_2d):
        patch_num = len(aImage_patches)
        total_patch_count += patch_num
        print('number of patches on Image %d: %d' % (img_idx,patch_num))
    print('total number of patches: %d'%total_patch_count)

    ##  inference image patches one by one, and save to disks
    for img_idx, aImage_patches in enumerate(data_patches_2d):
        print('start inference on Image  %d' % img_idx)
        instances_folder = 'I%d_patches_%d_%d_%d_%d' % (img_idx,patch_w, patch_h, overlay_x, overlay_y)
        json_dir = os.path.join(inf_output_dir,instances_folder)
        os.system('mkdir -p ' + json_dir)
        for idx, img_patch in enumerate(aImage_patches):

            # if not idx in [3]:
            #      continue

            img_data = build_RS_data.read_patch(img_patch)  # (nband, height,width)

            # test
            # img_save_path = "I%d_%d_org.tif" % (img_idx, idx)
            # build_RS_data.save_patch(img_patch, img_data, img_save_path)

            img_data = np.transpose(img_data, (1, 2, 0))  # keras and tf require (height,width,nband)
            # inference them
            results = model.detect([img_data], verbose=0)
            mrcc_r = results[0]

            print('Save Instances of Image:%d patch:%4d, shape:(%d,%d) to a json file' %
                  (img_idx, idx, img_patch.boundary[3], img_patch.boundary[2]))  # ysize, xsize

            # short the file name to avoid  error of " Argument list too long", hlc 2018-Oct-29
            file_name = "I%d_%d.txt" % (img_idx, idx)
            save_path = os.path.join(json_dir, file_name)
            if build_RS_data.save_instances_patch(img_patch,mrcc_r,save_path) is False:
                return False

    return True


def inf_remoteSensing_image(model,image_path=None):
    '''
    input a remote sensing image, then split to many small patches, inference each patch and merge than at last
    :param model: trained model
    :param image_path:
    :return: False if unsuccessful.
    '''

    # split images
    inf_image_dir=parameters.get_string_parameters(para_file,'inf_images_dir')

    patch_w = parameters.get_digit_parameters(para_file, "inf_patch_width", None, 'int')
    patch_h = parameters.get_digit_parameters(para_file, "inf_patch_height", None, 'int')
    overlay_x = parameters.get_digit_parameters(para_file, "inf_pixel_overlay_x", None, 'int')
    overlay_y = parameters.get_digit_parameters(para_file, "inf_pixel_overlay_y", None, 'int')


    inf_batch_size = parameters.get_digit_parameters(para_file, "inf_batch_size", None, 'int')

    global inf_list_file
    if image_path is not None:
        with open('inf_image_list.txt','w') as f_obj:
            f_obj.writelines(image_path)
            inf_list_file = 'inf_image_list.txt'

    data_patches_2d = build_RS_data.make_dataset(inf_image_dir,inf_list_file,
                patch_w,patch_h,overlay_x,overlay_y,train=False)

    if len(data_patches_2d)< 1:
        return False

    total_patch_count = 0
    for img_idx, aImage_patches in enumerate(data_patches_2d):
        patch_num = len(aImage_patches)
        total_patch_count += patch_num
        print('number of patches on Image %d: %d' % (img_idx,patch_num))
    print('total number of patches: %d'%total_patch_count)

    ##  inference multiple image patches at the same time
    # for img_idx, aImage_patches in enumerate(data_patches_2d):
    #
    #     print('start inference on Image  %d' % img_idx)
    #
    #     idx = 0     #index of all patches on this image
    #     patch_batches = build_RS_data.split_patches_into_batches(aImage_patches,inf_batch_size)
    #
    #     for a_batch_of_patches in patch_batches:
    #
    #         # Since it required a constant of batch size for the frozen graph, we copy (duplicate) the first patch
    #         org_patch_num = len(a_batch_of_patches)
    #         while len(a_batch_of_patches) < inf_batch_size:
    #             a_batch_of_patches.append(a_batch_of_patches[0])
    #
    #         # read image data and stack at 0 dimension
    #         multi_image_data = []
    #         for img_patch in a_batch_of_patches:
    #             img_data = build_RS_data.read_patch(img_patch) # (nband, height,width)
    #             img_data = np.transpose(img_data, (1, 2, 0))  # keras and tf require (height,width,nband)
    #             multi_image_data.append(img_data)
    #         # multi_images = np.stack(multi_image_data, axis=0)
    #
    #         # modify the BATCH_Size
    #         model.config.BATCH_SIZE = len(multi_image_data)
    #
    #         # inference them
    #         results = model.detect(multi_image_data, verbose=0)
    #         # r = results[0]
    #         # visualize.display_instances(image_data, r['rois'], r['masks'], r['class_ids'],
    #         #                             ['BG', 'thawslump'], r['scores'], ax=get_ax())
    #
    #         #save
    #         for num,(mrcc_r,img_patch) in enumerate(zip(results,a_batch_of_patches)):
    #
    #             # ignore the duplicated ones
    #             if num >= org_patch_num:
    #                 break
    #
    #
    #             # convert mask to a map of classification
    #             masks = mrcc_r['masks'] # shape: (height, width, num_classes?)
    #             height, width, nclass = masks.shape
    #
    #             seg_map = np.zeros((height, width),dtype=np.uint8)
    #             for n_id in range(0,nclass):
    #                 seg_map[ masks[:,:,n_id] == True ] = n_id + 1
    #
    #             print('Save segmentation result of Image:%d patch:%4d, shape:(%d,%d)' %
    #                   (img_idx, idx, seg_map.shape[0], seg_map.shape[1]))
    #
    #             # short the file name to avoid  error of " Argument list too long", hlc 2018-Oct-29
    #             file_name = "I%d_%d" % (img_idx, idx)
    #
    #             save_path = os.path.join(inf_output_dir, file_name + '.tif')
    #             if build_RS_data.save_patch_oneband_8bit(img_patch,seg_map.astype(np.uint8),save_path) is False:
    #                 return False
    #
    #             idx += 1

    #  inference image patches one by one
    for img_idx, aImage_patches in enumerate(data_patches_2d):

        print('start inference on Image  %d' % img_idx)

        for idx, img_patch in enumerate(aImage_patches):

            # # test debug
            # if not idx in [3359, 3360,3361,3476,3477,3478,3593,3594,3595]:
            #     continue
            # if not idx in [4700, 4817, 4818, 4819, 4934, 4935, 4936, 5051, 5052, 5053]:
            #     continue
            # if not idx in [2602]:  # a false positive
            #     continue


            img_data = build_RS_data.read_patch(img_patch)  # (nband, height,width)

            # test
            # save_path = "I%d_%d_org.tif" % (img_idx, idx)
            # build_RS_data.save_patch(img_patch, img_data, save_path)

            img_data = np.transpose(img_data, (1, 2, 0))  # keras and tf require (height,width,nband)

            # inference them
            results = model.detect([img_data], verbose=0)
            mrcc_r = results[0]

            # mrcc_r['scores']
            # mrcc_r['rois']
            masks = mrcc_r['masks']  # shape: (height, width, num_instance)
            height, width, ncount = masks.shape
            class_ids = mrcc_r['class_ids']

            seg_map = np.zeros((height, width), dtype=np.uint8)
            for inst in range(0, ncount):  # instance one by one
                seg_map[masks[:, :, inst] == True] = class_ids[inst]

            print('Save segmentation result of Image:%d patch:%4d, shape:(%d,%d)' %
                  (img_idx, idx, seg_map.shape[0], seg_map.shape[1]))

            # short the file name to avoid  error of " Argument list too long", hlc 2018-Oct-29
            file_name = "I%d_%d" % (img_idx, idx)

            save_path = os.path.join(inf_output_dir, file_name + '.tif')
            if build_RS_data.save_patch_oneband_8bit(img_patch, seg_map.astype(np.uint8), save_path) is False:
                return False



############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on Planet CubeSat images .')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train', 'evaluate', or 'inference' on Planet CubeSat images")
    # parser.add_argument('--dataset', required=True,
    #                     metavar="/path/to/planet/",
    #                     help='Directory of the Planet CubeSat images')
    parser.add_argument('--para_file', required=True,
                        default='para.ini',
                        metavar="para.ini",
                        help='the parameter file')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to init weights .h5 file or 'coco'")
    parser.add_argument('--inf_list_file',required=False,
                        default='inf_image_list.txt',
                        metavar='inf_image_list.txt',
                        help='a file contains lists of remote sensing images for inference'
                        )
    parser.add_argument('--inf_output_dir',required=False,
                        default='inf_results',
                        metavar='inf_results',
                        help='the folder to save image patches of inference results'
                        )
    # parser.add_argument('--logs', required=False,
    #                     default=DEFAULT_LOGS_DIR,
    #                     metavar="/path/to/logs/",
    #                     help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    # parser.add_argument('--download', required=False,
    #                     default=False,
    #                     metavar="<True|False>",
    #                     help='Automatically download and unzip MS-COCO files (default=False)',
    #                     type=bool)
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    # print("Dataset: ", args.dataset)
    print("the parameter file: ", args.para_file)

    # print("Logs: ", args.logs)
    # print("Auto Download: ", args.download)

    para_file = args.para_file
    inf_list_file = args.inf_list_file
    inf_output_dir = args.inf_output_dir


    expr_name = parameters.get_string_parameters(args.para_file, 'expr_name')
    if os.path.isdir(expr_name) is False:
        os.mkdir(expr_name)


    NO_DATA = parameters.get_digit_parameters(args.para_file, 'dst_nodata',None, 'int')
    num_class_noBG = parameters.get_digit_parameters(args.para_file,'NUM_CLASSES_noBG',None, 'int')

    # modify default setting according to different machine
    gpu_count = parameters.get_digit_parameters(args.para_file, 'gpu_count', None, 'int')
    images_per_gpu = parameters.get_digit_parameters(args.para_file, 'images_per_gpu', None, 'int')
    PlanetConfig.GPU_COUNT = gpu_count
    PlanetConfig.IMAGES_PER_GPU  = images_per_gpu

    # Number of training and validation steps per epoch, same as nucleus sample
    # https://github.com/matterport/Mask_RCNN/issues/1092
    # https://github.com/matterport/Mask_RCNN/issues/550
    train_file_count = sum(1 for line in open('list/train_list.txt'))
    val_file_count = sum(1 for line in open('list/val_list.txt'))
    batchsize = gpu_count*images_per_gpu
    PlanetConfig.STEPS_PER_EPOCH = train_file_count // batchsize+ 1
    PlanetConfig.VALIDATION_STEPS = max(1, val_file_count // batchsize)

    PlanetConfig.BACKBONE = parameters.get_string_parameters(args.para_file, 'BACKBONE')
    PlanetConfig.NUM_CLASSES = 1 + num_class_noBG


    # Which weights to start with?
    # init_with = "coco"  # imagenet, coco, or last

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(curr_dir,expr_name, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)


    # Directory to save logs and model checkpoints,
    logs_dir = os.path.join(curr_dir, expr_name)

    # ####################################################################################
    # # test OpenCV load mask
    # image_path = 'test_instance_masks/split_labels/20180522_035755_3B_AnalyticMS_SR_mosaic_8bit_rgb_basinExt_144_class_1_p_0.png'
    # image_data = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # height,width = image_data.shape
    #
    # # test_idx = image_data != 0
    # # image_data[image_data != 0] = 255
    # # cv2.imwrite('mask_255.tif', image_data)
    #
    # image, contours, hierarchy = cv2.findContours(image_data,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    #
    # # create seed mask
    # seed_masks = np.zeros((height,width),np.int32)
    # # for idx, contour in enumerate(contours):
    # #     # for point in contour:
    # #     #     print(point[0])
    # #     indexs = [point[0] for point in contour]
    # #     seed_masks[indexs] = idx +1
    # # cv2.drawContours(seed_masks, contours, -1, (255), 1) # should I used different color (255)
    # # cv2.drawContours(seed_masks, contours, -1, (255), -1)  #Negative thickness means that a filled circle is to be drawn.
    #
    # for idx in range(0,len(contours)):
    #     print('idx:',idx)
    #     cv2.drawContours(seed_masks, contours, idx, (idx +1), -1)
    #     point = contours[idx][0][0] # [col,row]
    #     print('point:',point)
    #     print('class_id:',image_data[point[1],point[0]])  # [row,col]
    #
    #
    # seed_masks = seed_masks * 50
    # cv2.imwrite('seed_masks_255.tif', seed_masks.astype(np.uint8))
    #
    # # get masks using
    # # markers = cv2.watershed(image_data, seed_masks)
    # # cv2.imwrite('instance_masks.tif', markers.astype(np.uint8))
    #
    # sys.exit(0)
    # pass

    ####################################################################################


    # Configurations
    if args.command == "train":
        config = PlanetConfig()
    else:
        class InferenceConfig(PlanetConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            NUM_CLASSES = 1 + num_class_noBG  # have the same class number
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=logs_dir)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=logs_dir)

    # Select weights file to load (init)
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    if args.model.lower() == "coco" and args.command == "train":
        model.load_weights(model_path, by_name=True, exclude = ["mrcnn_class_logits", "mrcnn_bbox_fc",
                   "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(model_path, by_name=True)


    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = PlanetDataset()
        dataset_train.load_Planet('split_images','split_labels','train')

        dataset_train.prepare()

        ## # Validation dataset
        dataset_val = PlanetDataset()
        dataset_val.load_Planet('split_images', 'split_labels', 'val')
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        # augmentation = imgaug.augmenters.Fliplr(0.5)
        # no_aug_sources = ['planet_noTS']

        # Image augmentation will be performed during preparing image patches
        augmentation = None
        no_aug_sources = None

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=10,  #40
                    layers='heads',
                    augmentation=augmentation,no_augmentation_sources=no_aug_sources)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=50, #120
                    layers='4+',
                    augmentation=augmentation,no_augmentation_sources=no_aug_sources)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=70, # 160
                    layers='all',
                    augmentation=augmentation,no_augmentation_sources=no_aug_sources)
    elif args.command == "evaluate":
        pass
    elif args.command == "inference":
        # inference on a small image (e.g., a patch on a RS image ) and visualize.

        ## randomly pick a image patch from validation subset
        # import matplotlib.pyplot as plt
        # def get_ax(rows=1, cols=1, size=8):
        #     """Return a Matplotlib Axes array to be used in
        #     all visualizations in the notebook. Provide a
        #     central point to control graph sizes.
        #
        #     Change the default size attribute to control the size
        #     of rendered images
        #     """
        #     _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
        #     return ax
        #
        # dataset_train = PlanetDataset()
        # dataset_train.load_Planet('split_images','split_labels','train')
        # dataset_train.prepare()
        #
        # ## # Validation dataset
        # dataset_val = PlanetDataset()
        # dataset_val.load_Planet('split_images', 'split_labels', 'val')
        # dataset_val.prepare()
        #
        # # Test on a random image
        # image_id = random.choice(dataset_val.image_ids)
        # print(dataset_val.image_reference(image_id))
        # original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val,
        #                                         config,image_id,use_mini_mask=False)
        #
        # log("original_image", original_image)
        # log("image_meta", image_meta)
        # log("gt_class_id", gt_class_id)
        # log("gt_bbox", gt_bbox)
        # log("gt_mask", gt_mask)
        #
        # visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
        #                             dataset_train.class_names, figsize=(8, 8))
        #
        # # In[13]:
        # # original_image is a numpy array (height, width, bands),
        # # its size is IMAGE_MIN_DIM by IMAGE_MAX_DIM when using modellib.load_image_gt
        # results = model.detect([original_image], verbose=1)
        #
        # r = results[0]
        # # r['masks'] have the same height and width of original_image
        # visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
        #                             dataset_val.class_names, r['scores'], ax=get_ax())


        # test one image:
        # img_path = "20180522_035755_3B_AnalyticMS_SR_mosaic_8bit_rgb_basinExt_37_class_1_p_4.png"
        # image_data = skimage.io.imread(os.path.join('split_images', img_path))

        img_path = "I0_3_org.tif"
        image_data = skimage.io.imread(img_path)

        results = model.detect([image_data], verbose=1)
        r = results[0]

        # test load instance from disks
        # r = build_RS_data.load_instances_patch(os.path.join(inf_output_dir,'I0_3.txt'),bDisplay=True)
        # r['rois'] = np.array(r['rois'])
        # r['class_ids'] = np.array(r['class_ids'])
        # r['scores'] = np.array(r['scores'])

        visualize.display_instances(image_data, r['rois'], r['masks'], r['class_ids'],
                                     ['BG','thawslump'], scores=r['scores'], figsize=(8, 8))
        log('rois:',r['rois'])
        log('masks:',r['masks'])
        log('class_ids:',r['class_ids'])
        log('scores',r['scores'])
        print('scores:',r['scores'])


        pass
    elif args.command == "inference_rsImg":
        # inference on RS images

        if not os.path.isdir(inf_output_dir):
            os.system('mkdir -p ' + inf_output_dir)
        inf_remoteSensing_image(model)

        pass

    elif args.command == "inference_rsImg_multi":
        # inference on RS images

        if not os.path.isdir(inf_output_dir):
            os.system('mkdir -p ' + inf_output_dir)
        muti_inf_remoteSensing_image(model)

        pass

    else:
        print("'{}' is not recognized. "
              "Use 'train', 'evaluate', 'inference', 'inference_rsImg', or 'inference_rsImg_multi' ".format(args.command))
