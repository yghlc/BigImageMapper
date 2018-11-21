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

# check python version
from distutils.version import LooseVersion
print(sys.version)
if (LooseVersion(sys.version) > LooseVersion('3.4')) is False:
    raise EnvironmentError('Require Python version > 3.4')

NO_DATA = 255
para_file = 'para_mrcnn.ini'
inf_list_file = 'saved_inf_list.txt'
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

    # We use a GPU with 12GB memory, which can fit two images for 1024*2014.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2 #??

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # only one class. i.e. thaw slumps

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
        self.add_class("planet", 1, "thawslump")    # source, class_id, class_name

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
            self.add_image("planet", image_id=i, path=image_path,
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

        # Load image
        label = skimage.io.imread(label_path)
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

        # create the mask for each class (excluding the background)
        for id, count in zip(unique_ids,counts):
            # ignore background
            if id==0:
                continue
            # Some objects are so small that they're less than 1 pixel area
            # and end up rounded out. Skip those objects.
            if count < 1:
                continue
            m = np.zeros([height, width], dtype=bool)

            m[label == id] = True

            instance_masks.append(m)
            class_ids.append(id)

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

    if image_path is not None:
        with open('saved_inf_list.txt','w') as f_obj:
            f_obj.writelines(image_path)
            inf_list_file = 'saved_inf_list.txt'

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

    for img_idx, aImage_patches in enumerate(data_patches_2d):

        print('start inference on Image  %d' % img_idx)

        idx = 0     #index of all patches on this image
        patch_batches = build_RS_data.split_patches_into_batches(aImage_patches,inf_batch_size)

        for a_batch_of_patches in patch_batches:

            # Since it required a constant of batch size for the frozen graph, we copy (duplicate) the first patch
            org_patch_num = len(a_batch_of_patches)
            while len(a_batch_of_patches) < inf_batch_size:
                a_batch_of_patches.append(a_batch_of_patches[0])

            # read image data and stack at 0 dimension
            multi_image_data = []
            for img_patch in a_batch_of_patches:
                img_data = build_RS_data.read_patch(img_patch)
                multi_image_data.append(img_data)
            # multi_images = np.stack(multi_image_data, axis=0)

            # inference them
            results = model.detect(multi_image_data, verbose=0)
            # r = results[0]
            # visualize.display_instances(image_data, r['rois'], r['masks'], r['class_ids'],
            #                             ['BG', 'thawslump'], r['scores'], ax=get_ax())

            #save
            for num,(mrcc_r,img_patch) in enumerate(zip(results,a_batch_of_patches)):

                # ignore the duplicated ones
                if num >= org_patch_num:
                    break


                # convert mask to a map of classification
                masks = mrcc_r['masks'] # shape: (height, width, num_classes?)
                height, width, nclass = masks.shape

                seg_map = np.zeros((height, width),dtype=np.uint8)
                for n_id in range(0,nclass):
                    seg_map[ masks[:,:,n_id] == True ] = n_id + 1

                print('Save segmentation result of Image:%d patch:%4d, shape:(%d,%d)' %
                      (img_idx, idx, seg_map.shape[0], seg_map.shape[1]))

                # short the file name to avoid  error of " Argument list too long", hlc 2018-Oct-29
                file_name = "I%d_%d" % (img_idx, idx)

                save_path = os.path.join(inf_output_dir, file_name + '.tif')
                if build_RS_data.save_patch_oneband_8bit(img_patch,seg_map.astype(np.uint8),save_path) is False:
                    return False

                idx += 1


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
                        default='saved_inf_list.txt',
                        metavar='saved_inf_list.txt',
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

    # modify default setting according to different machine
    gpu_count = parameters.get_digit_parameters(args.para_file, 'gpu_count', None, 'int')
    images_per_gpu = parameters.get_digit_parameters(args.para_file, 'images_per_gpu', None, 'int')
    PlanetConfig.GPU_COUNT = gpu_count
    PlanetConfig.IMAGES_PER_GPU  = images_per_gpu


    # Which weights to start with?
    # init_with = "coco"  # imagenet, coco, or last

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(curr_dir,expr_name, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)


    # Directory to save logs and model checkpoints,
    logs_dir = os.path.join(curr_dir, expr_name)


    # Configurations
    if args.command == "train":
        config = PlanetConfig()
    else:
        class InferenceConfig(PlanetConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            NUM_CLASSES = 2  # have the same class number
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
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+',
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all',
                    augmentation=augmentation)
    elif args.command == "evaluate":
        pass
    elif args.command == "inference":
        # inference on a small image (e.g., a patch on a RS image ) and visualize.

        ## randomly pick a image patch from validation subset
        import matplotlib.pyplot as plt
        def get_ax(rows=1, cols=1, size=8):
            """Return a Matplotlib Axes array to be used in
            all visualizations in the notebook. Provide a
            central point to control graph sizes.

            Change the default size attribute to control the size
            of rendered images
            """
            _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
            return ax
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
        img_path = "20180522_035755_3B_AnalyticMS_SR_mosaic_8bit_rgb_basinExt_37_class_1_p_4.png"
        # Load image
        image_data = skimage.io.imread(os.path.join('split_images',img_path))
        results = model.detect([image_data], verbose=1)
        r = results[0]
        visualize.display_instances(image_data, r['rois'], r['masks'], r['class_ids'],
                                     ['BG','thawslump'], r['scores'], ax=get_ax())
        log('rois:',r['rois'])
        log('masks:',r['masks'])
        log('class_ids:',r['class_ids'])
        log('scores',r['scores'])


        pass
    elif args.command == "inference_rsImg":
        # inference on a RS image

        if not os.path.isdir(inf_output_dir):
            os.system('mkdir -p ' + inf_output_dir)
        inf_remoteSensing_image(model)

        pass

    else:
        print("'{}' is not recognized. "
              "Use 'train', 'evaluate', 'inference', or 'inference_rsImg' ".format(args.command))
