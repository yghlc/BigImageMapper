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

# import skimage
import skimage.io
import skimage.color

# check python version
from distutils.version import LooseVersion
print(sys.version)
if (LooseVersion(sys.version) > LooseVersion('3.4')) is False:
    raise EnvironmentError('Require Python version > 3.4')



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

# path of DeeplabforRS
codes_dir2 = HOME +'/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)

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
    # GPU_COUNT = 8

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

        height, width = label.shape
        unique_ids, counts = np.unique(label, return_counts=True)

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
                        help="'train' or 'evaluate' on Planet CubeSat images")
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

    expr_name = parameters.get_string_parameters(args.para_file, 'expr_name')
    if os.path.isdir(expr_name) is False:
        os.mkdir(expr_name)

    # Which weights to start with?
    # init_with = "coco"  # imagenet, coco, or last
    # Path to trained weights file
    # PLANET_MODEL_PATH = os.path.join(curr_dir,expr_name, "mask_rcnn_planet.h5")



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
    model.load_weights(model_path, by_name=True, exclude = ["mrcnn_class_logits", "mrcnn_bbox_fc",
               "mrcnn_bbox", "mrcnn_mask"])

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = PlanetDataset()
        dataset_train.load_Planet('split_images','split_labels','train')

        dataset_train.prepare()

        ## # Validation dataset
        dataset_val = PlanetDataset()
        dataset_train.load_Planet('split_images', 'split_labels', 'val')
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
        'inference'
        pass

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate' ".format(args.command))
