#!/usr/bin/env python
# Filename: image_classify_rnn 
"""
introduction: classification multi-spectral remote sensing images using RNN. This is pixel-based classification

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 10 April, 2018
"""


import sys,os
from optparse import OptionParser

import numpy as np
import random
import rasterio


###-------------------------- set gpu using tf ---------------------------
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
###-------------------  start importing keras module ---------------------

import keras
from keras.models import Sequential
from keras.layers import Input, Dense, TimeDistributed
from keras.models import Model
from keras.layers import LSTM

num_classes = 20
hidden_units = 128
batch_size = 128

epochs = 200

def read_oneband_image_to_1dArray(image_path):

    if os.path.isfile(image_path) is False:
        print("error, file not exist: " + image_path)
        return None

    with rasterio.open(image_path) as img_obj:
        # read the all bands (only have one band)
        indexes = img_obj.indexes
        if len(indexes) != 1:
            print('error, only support one band')
            return None

        data = img_obj.read(indexes)

        data_1d = data.flatten()  # convert to one 1d, row first.

        return data_1d


def read_multiband_image_to_2dArray(image_path):
    """

    :param image_path:
    :return: 2d Array (bands, number of pixels)
    """

    if os.path.isfile(image_path) is False:
        print("error, file not exist: " + image_path)
        return None

    with rasterio.open(image_path) as img_obj:
        # read the all bands (only have one band)
        indexes = img_obj.indexes

        data = img_obj.read(indexes)
        band,height, width = data.shape
        # print(data.shape)
        data = np.transpose(data,(1,2,0))  # this is necessary before reshape, or the arrary is wrong
        data_2d = data.reshape(height*width,band)  # row first
        return data_2d

def split_data(x_all, y_all, test_percent=0.01):
    """
    split the data to train set and test set
    :param x_all: all the x : (count,n_features)
    :param y_all: all the y : (count,1)
    :param percent: the percent of the test set [0,1]
    :return: (x_train, y_train), (x_test, y_test)
    """

    total_count = x_all.shape[0]
    test_count = int(total_count*test_percent)

    # random select the test sample
    # bug: np.random.randint could output some duplicated number, which causes not consist amount when using np.delete
    # test_index = np.random.randint(0,total_count,size=test_count)
    test_index = random.sample(range(total_count), test_count)
    # test_index = np.array(range(0,test_count))
    # test_index = sorted(test_index)

    x_test = x_all[test_index]
    y_test = y_all[test_index]
    # print(len(test_index), min(test_index),max(test_index),'size, minimum, and maximum of of test_index')

    x_train = np.delete(x_all,test_index,axis=0)
    y_train = np.delete(y_all,test_index,axis=0)

    # print(x_all.shape[0], 'total count before splitting')
    # print(x_train.shape[0]+x_test.shape[0],'total count after splitting')
    #
    # print(x_train.shape[0],y_train.shape[0], 'train samples (x,y)')
    # print(x_test.shape[0],y_test.shape[0], 'test samples (x,y)')

    return (x_train, y_train), (x_test, y_test)

def build_train_rnn_model():

    model = Sequential()
    model.add(LSTM(hidden_units))
    model.add(Dense(num_classes, activation='sigmoid'))

    return model


def main(options, args):

    label_image = args[0]
    multi_spec_image_path = args[1]

    # read images
    label_1d = read_oneband_image_to_1dArray(label_image)
    multiBand_value_2d = read_multiband_image_to_2dArray(multi_spec_image_path)

    pixel_count = label_1d.shape[0]
    print(label_1d.shape,multiBand_value_2d.shape)

    # print ten pixels for checking
    # for i in range(10):
    #     index = random.randint(1,label_1d.shape[0])
    #     row = index/2384    # 2384 is the width of the input image
    #     col = index%2384
    #     print("row: %d, col: %d, label: %d"%(row,col,label_1d[index]))
    #     print("pixel value: "+ str(multiBand_value_2d[index]))

    # remove the non-ground truth pixels, that is "0" pixel
    back_ground_index = np.where(label_1d==0)
    label_1d = np.delete(label_1d,back_ground_index)
    multiBand_value_2d = np.delete(multiBand_value_2d, back_ground_index,axis=0)

    print("%.2f %% are unclassified (no observation)"%(len(back_ground_index[0])*100.0/pixel_count))
    print('after removing non-ground truth pixels',label_1d.shape, multiBand_value_2d.shape)

    # split train and test dataset
    (x_train, y_train), (x_test, y_test) = split_data(multiBand_value_2d, label_1d, test_percent=0.1)
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0],y_train.shape[0], 'train samples')
    print(x_test.shape[0],x_test.shape[0], 'test samples')

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # the original data is UINT16, the maximum value is around 45048 for this dataset, but use the simple way here
    x_train /= 65536
    x_test /= 65536

    bands = x_train.shape[1:]

    # 2D input.
    x = Input(shape=(bands))

    model = build_train_rnn_model()

    # Training.
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    # Evaluation.
    # verbose: 0 or 1. Verbosity mode. 0 = silent, 1 = progress bar.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])








if __name__ == "__main__":
    usage = "usage: %prog [options] label_image multi_spectral_images"
    parser = OptionParser(usage=usage, version="1.0 2018-4-10")
    parser.description = 'Introduction: classification multi-spectral remote sensing images using RNN '

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    # if options.para_file is None:
    #     basic.outputlogMessage('error, parameter file is required')
    #     sys.exit(2)

    main(options, args)