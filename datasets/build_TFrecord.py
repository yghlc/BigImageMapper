# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# modify from build_voc2012_data.py

# for more information on pascal_voc_seg: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#data
# voc2012, has 20 classes, label from 1 to 20. background has label 0.
# label 255 are boundaries and will be ignore in DeepLab

"""Converts PASCAL VOC 2012 data to TFRecord file format with Example protos.

PASCAL VOC 2012 dataset is expected to have the following directory structure:

  + pascal_voc_seg
    - build_data.py
    - build_voc2012_data.py (current working directory).
    + VOCdevkit
      + VOC2012
        + JPEGImages
        + SegmentationClass
        + ImageSets
          + Segmentation
    + tfrecord

Image folder:
  ./VOCdevkit/VOC2012/JPEGImages

Semantic segmentation annotations:
  ./VOCdevkit/VOC2012/SegmentationClass

list folder:
  ./VOCdevkit/VOC2012/ImageSets/Segmentation

This script converts data into sharded data files and save at tfrecord folder.

The Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded semantic segmentation content.
  image/segmentation/class/format: semantic segmentation file format.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os.path
import sys
import build_data
from six.moves import range
import tensorflow as tf
# import tensorflow.compat.v1 as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_folder',
                           './split_images',
                           'Folder containing images.')

tf.app.flags.DEFINE_string(
    'semantic_segmentation_folder',
    './split_labels',
    'Folder containing semantic segmentation annotations.')

tf.app.flags.DEFINE_string(
    'list_folder',
    './list',
    'Folder containing lists for training and validation')

tf.app.flags.DEFINE_string(
    'output_dir',
    './tfrecord',
    'Path to save converted SSTable of TensorFlow examples.')


_NUM_SHARDS = 4

para_file = sys.argv[1]
if os.path.isfile(para_file) is False:
    raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

code_dir = os.path.join(os.path.dirname(sys.argv[0]), '..')
sys.path.insert(0, code_dir)
import parameters


def _convert_dataset(dataset_split):
    """Converts the specified dataset split to TFRecord format.

    Args:
    dataset_split: The dataset split (e.g., train, test).

    Raises:
    RuntimeError: If loaded image and label have different shape.
    """
    dataset = os.path.basename(dataset_split)[:-4]
    sys.stdout.write('Processing ' + dataset)
    filenames = [x.strip('\n') for x in open(dataset_split, 'r')]
    num_images = len(filenames)
    num_per_shard = int(math.ceil(num_images / _NUM_SHARDS))

    # image_reader = build_data.ImageReader('jpeg', channels=3)
    image_reader = build_data.ImageReader('png', channels=3)
    label_reader = build_data.ImageReader('png', channels=1)

    os.system("mkdir -p " + FLAGS.output_dir)

    for shard_id in range(_NUM_SHARDS):
        output_filename = os.path.join(
            FLAGS.output_dir,
            '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_idx = shard_id * num_per_shard
          end_idx = min((shard_id + 1) * num_per_shard, num_images)
          for i in range(start_idx, end_idx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i + 1, len(filenames), shard_id))
            sys.stdout.flush()
            # Read the image.
            image_filename = os.path.join(
                FLAGS.image_folder, filenames[i] + '.' + FLAGS.image_format)
            image_data = tf.gfile.GFile(image_filename, 'rb').read()
            height, width = image_reader.read_image_dims(image_data)
            # Read the semantic segmentation annotation.
            seg_filename = os.path.join(
                FLAGS.semantic_segmentation_folder,
                filenames[i] + '.' + FLAGS.label_format)
            seg_data = tf.gfile.GFile(seg_filename, 'rb').read()
            seg_height, seg_width = label_reader.read_image_dims(seg_data)
            if height != seg_height or width != seg_width:
              raise RuntimeError('Shape mismatched between image and label.')
            # Convert to tf example.
            example = build_data.image_seg_to_tfexample(
                image_data, filenames[i], height, width, seg_data)
            tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()


def main(unused_argv):
    train_sample_txt = parameters.get_string_parameters_None_if_absence(para_file,'training_sample_list_txt')
    val_sample_txt = parameters.get_string_parameters_None_if_absence(para_file,'validation_sample_list_txt')
    if train_sample_txt is not None and val_sample_txt is not None:
        train_sample_txt = os.path.join(FLAGS.list_folder, train_sample_txt)
        val_sample_txt = os.path.join(FLAGS.list_folder, val_sample_txt)
    else:
        raise ValueError('training_sample_list_txt or validation_sample_list_txt are not in %s'%para_file)

    if os.path.isfile(train_sample_txt) is False:
        raise IOError('%s does not exist'%train_sample_txt)
    if os.path.isfile(val_sample_txt) is False:
        raise IOError('%s does not exist'%val_sample_txt)

    dataset_splits = [train_sample_txt,val_sample_txt]
    # dataset_splits = tf.gfile.Glob(os.path.join(FLAGS.list_folder, '*val.txt'))
    print(dataset_splits)


    for dataset_split in dataset_splits:
        _convert_dataset(dataset_split)


if __name__ == '__main__':
    tf.app.run()
