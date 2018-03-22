
# inference a remote sensing images (need to split the images first, then merge the results)
# modified from "tensorflow/models/research/deeplab/deeplab_demo.ipynb"

# coding: utf-8

# # DeepLab Demo
# 
# This demo will demonstrate the steps to run deeplab semantic segmentation model on sample input images.
# 
# ## Prerequisites
# 
# Running this demo requires the following libraries:
# 
# * Jupyter notebook (Python 2)
# * Tensorflow (>= v1.5.0)
# * Matplotlib
# * Pillow
# * numpy

import os
import sys

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image


HOME = os.path.expanduser('~')
basicCodes_path = HOME + '/codes/PycharmProjects/DeeplabforRS'
sys.path.append(basicCodes_path)

import parameters

# set GPU on Cryo06, it seem this code not works
#os.system("export CUDA_VISIBLE_DEVICES=0")

import tensorflow as tf

if tf.__version__ < '1.5.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.5.0 or newer!')


tf_research_dir="/home/hlc/codes/PycharmProjects/tensorflow/models/research"
deeplab_dir=os.path.join(tf_research_dir,"deeplab")

# current folder, usually is where I run the codes
WORK_DIR=os.getcwd()
expr_name=parameters.get_string_parameters('para.ini','expr_name')

# Needed to show segmentation colormap labels
sys.path.append(os.path.join(deeplab_dir,'utils'))
import get_dataset_colormap


# ## Select and download models
# ## Load model in TensorFlow

_FROZEN_GRAPH_NAME = 'frozen_inference_graph'
class DeepLabModel(object):
    """Class to load deeplab model and run inference."""
    
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513

    def __init__(self, frozen_graph_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        with open(frozen_graph_path, "rb") as f:
            graph_def = tf.GraphDef.FromString(f.read())

        if graph_def is None:
            raise RuntimeError('Error, Cannot Open inference graph.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.
        
        Args:
            image: A PIL.Image object, raw input image.
            
        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map



# ## Helper methods

# LABEL_NAMES = np.asarray([
#     'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
#     'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
#     'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
#     'train', 'tv'
# ])

LABEL_NAMES = np.asarray([
    'Unclassified', 'Healthy grass', 'Stressed grass', 'Artificial turf', 'Evergreen trees', 'Deciduous trees',
    'Bare earth', 'Water', 'Residential buildings', 'Non-residential buildings', 'Roads', 'Sidewalks', 'Crosswalks',
    'Major thoroughfares', 'Highways', 'Railways', 'Paved parking lots', ' Unpaved parking lots', 'Cars',
    'Trains', 'Stadium seats'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = get_dataset_colormap.label_to_color_image(FULL_LABEL_MAP)


def vis_segmentation(image, seg_map):
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')
    
    plt.subplot(grid_spec[1])
    seg_image = get_dataset_colormap.label_to_color_image(
        seg_map, get_dataset_colormap.get_pascal_name()).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')
    
    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0)
    plt.show()


# ## Run on sample images
# Note that we are using single scale inference in the demo for fast
# computation, so the results may slightly differ from the visualizations
# in README, which uses multi-scale and left-right flipped inputs.

IMAGE_DIR = os.path.join(WORK_DIR,'split_images')

def run_demo_image(model,image_name):
    try:
        image_path = os.path.join(IMAGE_DIR, image_name)
        orignal_im = Image.open(image_path)
    except IOError:
        print 'Failed to read image from %s.' % image_path 
        return 
    print 'running deeplab on image %s...' % image_name
    resized_im, seg_map = model.run(orignal_im)
    
    vis_segmentation(resized_im, seg_map)



def main(unused_argv):

    # model = DeepLabModel(download_path) # this input a tarball
    frozen_graph_path = os.path.join(WORK_DIR, expr_name, 'export', 'frozen_inference_graph.pb')
    if os.path.isfile(frozen_graph_path) is False:
        raise RuntimeError('the file of inference graph is not exist, file path:' + frozen_graph_path)
    model = DeepLabModel(frozen_graph_path)

    image_name = ['UH17_GI1F051_TR_8bit_p_0.png', 'UH17_GI1F051_TR_8bit_p_6.png', 'UH17_GI1F051_TR_8bit_p_14.png']
    for image in image_name:
        run_demo_image(model,image)

if __name__ == '__main__':
    tf.app.run()





