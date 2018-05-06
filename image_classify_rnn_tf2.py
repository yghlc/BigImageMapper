#!/usr/bin/env python
# Filename: image_classify_rnn 
"""
introduction: classification multi-spectral remote sensing images using RNN. This is pixel-based classification

using tensorflow to implement this ideas

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

from tensorflow.contrib import rnn


import matplotlib.pyplot as plt
import datetime
import time

import sklearn

#we only have 20 classes, however, class index start from 1, so use 21. We should ignore the "0" class in the end
num_classes = 21
learn_rate = 0.001
hidden_units = 128
batch_size = 2048
lstm_layer_num = 1

epoches = 300

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

# def build_train_rnn_model(x_shape):
#
#     model = tf.keras.models.Sequential()
#     # model.add(LSTM(hidden_units, input_shape=x_shape))
#     model.add(tf.keras.layers.LSTM(hidden_units,input_shape=x_shape,return_sequences=True)) #,return_sequences=True
#
#     model.add(tf.keras.layers.LSTM(hidden_units, return_sequences=True))
#     model.add(tf.keras.layers.LSTM(hidden_units, return_sequences=True))
#     model.add(tf.keras.layers.LSTM(hidden_units, return_sequences=True))
#
#     model.add(tf.keras.layers.LSTM(hidden_units))
#
#     model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
#
#     # complie model
#     model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
#
#     return model

def lstm_layer(hidden_units,keep_prob):
    """
    setting a lstm layer
    :param hidden_units: the number of hidden units in this layer
    :param keep_prob: the dropout parameters, i.e. how many percent want to keep
    :return: this layer
    """

    layer = tf.contrib.rnn.BasicLSTMCell(hidden_units)
    # layer = tf.contrib.rnn.LSTMCell(hidden_units)

    # dropout, only apply to the output units
    cell = tf.contrib.rnn.DropoutWrapper(layer, output_keep_prob=keep_prob)
    return layer

def multi_lstm_layers(hidden_units,keep_prob,layer_num):

    mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_layer(hidden_units, keep_prob) for _ in range(layer_num)],
                                             state_is_tuple=True)
    return mlstm_cell

def main(options, args):

    t0 = time.time()

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

    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    y_train = tf.keras.utils.to_categorical(y_train,num_classes)
    y_test = tf.keras.utils.to_categorical(y_test,num_classes)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0],y_train.shape[0], 'train samples')
    print(x_test.shape[0],x_test.shape[0], 'test samples')

    train_samples_num = x_train.shape[0]
    test_samples_num = x_test.shape[0]

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # the original data is UINT16, the maximum value is around 45048 for this dataset, but use the simple way here
    x_train /= 65536
    x_test /= 65536

    # bands is the same as feature number of this multi-spectral images
    bands = list(x_train.shape[1:])
    band_num = x_train.shape[1]

    ##############################################################
    ## start tensorflow codes here
    num_units = hidden_units
    n_classes = num_classes

    time_steps = band_num
    n_input = 1
    learning_rate = learn_rate

    # batch size can be changed during the last step in one epoch
    batch_size_v = tf.placeholder(tf.int32, [])
    # keep_prob_v = tf.placeholder(tf.float32, [])   # keep_prob_v is different when training and prediction

    # weights and biases of appropriate shape to accomplish above task
    out_weights = tf.Variable(tf.random_normal([num_units, n_classes]))
    out_bias = tf.Variable(tf.random_normal([n_classes]))

    # defining placeholders
    # input image placeholder
    x = tf.placeholder("float", [None, time_steps, n_input])
    # input label placeholder
    y = tf.placeholder("float", [None, n_classes])

    # processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
    input = tf.unstack(x, time_steps, 1)

    # defining the network
    lstm_layer = rnn.BasicLSTMCell(num_units, forget_bias=1)
    outputs, _ = rnn.static_rnn(lstm_layer, input, dtype="float32")

    # converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
    prediction = tf.matmul(outputs[-1], out_weights) + out_bias

    # loss_function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # optimization
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # model evaluation
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # initialize variables
    init = tf.global_variables_initializer()
    # train_dataset = tf.data.Dataset.zip((x_train,y_train))
    # train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
    # train_dataset = train_dataset.batch(batch_size)
    #
    # train_dataset = train_dataset.repeat(epoches)
    # iterator = train_dataset.make_one_shot_iterator()

    # with tf.Session() as sess:
    #     sess.run(init)
    #     iter = 1
    #
    #     iter_per_epoch = train_samples_num/batch_size
    #
    #     while iter < 10000:
    #         # batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
    #         # batch_x, batch_y = iterator.get_next()
    #
    #         # batch_x = batch_x.reshape((batch_size, time_steps, n_input))
    #         # batch_x = sess.run(batch_x)
    #         # batch_y = sess.run(batch_y)
    #
    #         read_iter = iter%iter_per_epoch
    #         batch_x = x_train[read_iter*batch_size:(read_iter+1)*batch_size]
    #         batch_y = y_train[read_iter * batch_size:(read_iter + 1) * batch_size]
    #
    #
    #         sess.run(opt, feed_dict={x: batch_x, y: batch_y})
    #
    #         if iter % 10 == 0:
    #             acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
    #             los = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
    #             print("For iter ", iter)
    #             print("Accuracy ", acc)
    #             print("Loss ", los)
    #             print("__________________")
    #
    #         iter = iter + 1
    #
    #     # calculating test accuracy
    #     test_data = x_test
    #     test_label = y_test
    #     print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))



    session.run(tf.global_variables_initializer())

    history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    f_obj = open("%s_loss.txt" % datetime_str, "w")

    for epoch in range(epoches):
        t_epoch = time.time()
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0

        # shuffle train data
        x_train, y_train = sklearn.utils.shuffle(x_train, y_train)

        batch_num = train_samples_num / batch_size
        batch_last_size = train_samples_num % batch_size
        if batch_last_size > 0:
            batch_num += 1
        # train
        for iter in range(batch_num):
            temp_batch_size = batch_size
            if iter == batch_num - 1:
                temp_batch_size = batch_last_size
                X_batch = x_train[iter * batch_size:]
                y_batch = y_train[iter * batch_size:]
                continue
                # print(X_batch.shape,y_batch.shape)
            else:
                X_batch = x_train[iter * batch_size:(iter + 1) * batch_size]
                y_batch = y_train[iter * batch_size:(iter + 1) * batch_size]
                # print(X_batch.shape, y_batch.shape)

            cost, acc, _ = session.run([loss, accuracy, opt],
                                       feed_dict={x: X_batch, y: y_batch,
                                                  batch_size_v: temp_batch_size})
            train_loss += cost
            train_acc += acc
            print("iter {}, train loss={:.6f}, acc={:.6f}".format(iter + 1, cost, acc))

        train_loss /= batch_num
        train_acc /= batch_num
        # test
        X_batch, y_batch = x_test, y_test

        _cost, _acc = session.run([loss, accuracy],
                                  feed_dict={x: X_batch, y: y_batch,
                                             batch_size_v: test_samples_num})
        val_acc = _acc
        val_loss = _cost
        out_str = "epoch {}, train loss={:.6f}, acc={:.6f}; test loss={:.6f}, acc={:.6f}; time cost: {:.2f} seconds". \
            format(epoch + 1, train_loss, train_acc, val_loss, val_acc, time.time() - t_epoch)
        print(out_str)
        f_obj.writelines(out_str + '\n')
        f_obj.flush()

        history['acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)

    f_obj.close()

    # list all data in history
    print(history.keys())

    # summarize history for accuracy
    plt.figure(0)
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    plt.savefig('%s_acc_his_%d.png' % (datetime_str, random.randint(1, 1000)), dpi=300)
    # summarize history for loss
    plt.figure(1)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    plt.savefig('%s_loss_his_%d.png' % (datetime_str, random.randint(1, 1000)), dpi=300)

    t1 = time.time()
    total = t1 - t0
    print(
    'complete, total time cost: %.2f seconds or %.2f minutes or %.2f hours' % (total, total / 60.0, total / 3600.0))



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