
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

landuse_path = HOME + '/codes/PycharmProjects/Landuse_DL'
sys.path.append(landuse_path)
import datasets.build_RS_data as build_RS_data

# set GPU on Cryo06, it seem this code not works
#os.system("export CUDA_VISIBLE_DEVICES=0")

import tensorflow as tf

if tf.__version__ < '1.5.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.5.0 or newer!')

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'inf_para_file',
    'para.ini',
    'The parameter file containing file path and parameters')

tf.app.flags.DEFINE_string(
    'inf_list_file',
    'inf_image_list.txt',
    'file containing lists for remote sensing and label images')

tf.app.flags.DEFINE_integer(
    'batch_size',
    1,
    'the number of each patches input to the model each time')

tf.app.flags.DEFINE_string(
    'inf_output_dir',
    './inf_results',
    'Path to save inference results.')

tf.app.flags.DEFINE_string(
    'frozen_graph',
    'frozen_inference_graph.pb',
    'File name of frozen inference graph')

tf_research_dir="/home/hlc/codes/PycharmProjects/tensorflow/models/research"
deeplab_dir=os.path.join(tf_research_dir,"deeplab")

# current folder, usually is where I run the codes
WORK_DIR=os.getcwd()
expr_name=parameters.get_string_parameters(FLAGS.inf_para_file,'expr_name')

# Needed to show segmentation colormap labels
sys.path.append(os.path.join(deeplab_dir,'utils'))
import get_dataset_colormap


# ## Select and download models
# ## Load model in TensorFlow

# _FROZEN_GRAPH_NAME = 'frozen_inference_graph'

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

        # print(graph_def)

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

    def run_rsImg_multi_patches(self, multi_images):
        """Runs inference on multiple patches of a remote sensing image.

        Args:
            image: A numpy array of multiple patches, multi_images has size (count, width,size, band), e.g. (10,480,480,3)

        Returns:
            seg_map: multiple seg_map
        """

        ## error: ValueError: Cannot feed value of shape (2, 480, 320, 3) for Tensor u'ImageTensor:0', which has shape '(1, ?, ?, 3)'
        ## it turns out that self.INPUT_TENSOR_NAME only accept one image each time.  Oct 30 2018.

        # input shape is '(ncount, ?, ?, 3)'
        # image_tran = np.transpose(multi_images,(0,2,3,1))
        image_tran_list = [np.transpose(image,(1,2,0)) for image in multi_images ]
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: image_tran_list})
        # seg_map = batch_seg_map[0]
        return batch_seg_map
        # return seg_map

    def run_rsImg_patch(self, image):
        """Runs inference on a patch of remote sensing images.

        Args:
            image: A numpy array of the patch

        Returns:
            seg_map: Segmentation map of `resized_image`.
        """
        # width, height = image.size
        # print(image.shape)
        # bands, height, width = image.shape
        # resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        # target_size = (int(resize_ratio * width), int(resize_ratio * height))
        # resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)

        # input shape is '(1, ?, ?, 3)'
        image_tran = np.transpose(image,(1,2,0))
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [image_tran]})
        seg_map = batch_seg_map[0]
        return seg_map


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


def inf_remoteSensing_image(model,image_path=None):
    '''
    input a remote sensing image, then split to many small patches, inference each patch and merge than at last
    :param model: trained model 
    :param image_path: 
    :return: False if unsuccessful.
    '''

    # split images
    inf_image_dir=parameters.get_string_parameters(FLAGS.inf_para_file,'inf_images_dir')

    patch_w = parameters.get_digit_parameters(FLAGS.inf_para_file, "inf_patch_width", None, 'int')
    patch_h = parameters.get_digit_parameters(FLAGS.inf_para_file, "inf_patch_height", None, 'int')
    overlay_x = parameters.get_digit_parameters(FLAGS.inf_para_file, "inf_pixel_overlay_x", None, 'int')
    overlay_y = parameters.get_digit_parameters(FLAGS.inf_para_file, "inf_pixel_overlay_y", None, 'int')

    if image_path is not None:
        with open('saved_inf_list.txt','w') as f_obj:
            f_obj.writelines(image_path)
            FLAGS.inf_list_file = 'saved_inf_list.txt'

    data_patches_2d = build_RS_data.make_dataset(inf_image_dir,FLAGS.inf_list_file,
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

        # split to many batches (groups)
        idx = 0     #index of all patches on this image
        patch_batches = build_RS_data.split_patches_into_batches(aImage_patches,FLAGS.batch_size)
        for a_batch_of_patches in patch_batches:

            # read image data and stack at 0 dimension
            multi_image_data = []
            for img_patch in a_batch_of_patches:
                img_data = build_RS_data.read_patch(img_patch)
                multi_image_data.append(img_data)
            # multi_images = np.stack(multi_image_data, axis=0)

            # inference them
            a_batch_seg_map = model.run_rsImg_multi_patches(multi_image_data)

            #save
            for (seg_map,img_patch) in zip(a_batch_seg_map,a_batch_of_patches):
                print('Save segmentation result of Image:%d patch:%4d, shape:(%d,%d)' %
                      (img_idx, idx, seg_map.shape[0], seg_map.shape[1]))

                # short the file name to avoid  error of " Argument list too long", hlc 2018-Oct-29
                file_name = "I%d_%d" % (img_idx, idx)

                save_path = os.path.join(FLAGS.inf_output_dir, file_name + '.tif')
                if build_RS_data.save_patch_oneband_8bit(img_patch,seg_map.astype(np.uint8),save_path) is False:
                    return False

                idx += 1

        # for (idx,img_patch) in enumerate(aImage_patches):
        #
        #     org_img = img_patch.org_img
        #
        #     # img_name_noext = os.path.splitext(os.path.basename(img_patch.org_img))[0]+'_'+str(idx)
        #
        #     # get segmentation map
        #     # each patch should not exceed INPUT_SIZE(513), or it will be resized.
        #     img_data = build_RS_data.read_patch(img_patch)
        #     print('inference at Image:%d patch:%4d, shape:(%d,%d,%d)'%(img_idx,idx,img_data.shape[0],img_data.shape[1],img_data.shape[2]))
        #
        #     # img = Image.fromarray(np.transpose(img_data,(1,2,0)), 'RGB')
        #     # img.save('test_readpatch_before_run.png')
        #
        #     seg_map = model.run_rsImg_patch(img_data)
        #
        #     # img = Image.fromarray(np.transpose(img_data,(1,2,0)), 'RGB')
        #     # img.save('test_readpatch.png')
        #
        #     # save segmentation map
        #     # file_name = os.path.splitext(os.path.basename(org_img))[0] + '_' + str(idx)+'_pred'
        #     file_name = "I%d_%d"%(img_idx,idx) # short the file name to avoid  error of " Argument list too long", hlc 2018-Oct-29
        #
        #     # print(file_name)
        #     save_path = os.path.join(FLAGS.inf_output_dir,file_name+'.tif')
        #     if build_RS_data.save_patch_oneband_8bit(img_patch,seg_map.astype(np.uint8),save_path) is False:
        #         return False



def main(unused_argv):

    # model = DeepLabModel(download_path) # this input a tarball
    frozen_graph_path = os.path.join(WORK_DIR, expr_name, 'export', FLAGS.frozen_graph)
    if os.path.isfile(frozen_graph_path) is False:
        raise RuntimeError('the file of inference graph is not exist, file path:' + frozen_graph_path)
    model = DeepLabModel(frozen_graph_path)

    # image_name = ['UH17_GI1F051_TR_8bit_p_0.png', 'UH17_GI1F051_TR_8bit_p_6.png', 'UH17_GI1F051_TR_8bit_p_14.png']
    # for image in image_name:
    #     run_demo_image(model,image)

    os.system('mkdir -p ' + FLAGS.inf_output_dir)
    inf_remoteSensing_image(model)

if __name__ == '__main__':
    tf.app.run()





