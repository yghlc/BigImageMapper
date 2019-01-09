#!/usr/bin/env python
# Filename: planet_svm_classify
"""
introduction: Using SVM in sklearn library to perform classification on Planet images

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 4 January, 2019
"""

import sys, os
from optparse import OptionParser

import rasterio
import numpy as np

HOME = os.path.expanduser('~')
# Landuse_DL
codes_dir = HOME + '/codes/PycharmProjects/Landuse_DL'
sys.path.insert(0, codes_dir)
sys.path.insert(0, os.path.join(codes_dir, 'datasets'))

# path of DeeplabforRS
codes_dir2 = HOME + '/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)

import basic_src.io_function as io_function
import basic_src.basic as basic

# Preprocessing
from sklearn import preprocessing
# library for SVM classifier
from sklearn import svm

# model_selection  # change grid_search to model_selection
from sklearn import model_selection

from sklearn.externals import joblib  # save and load model

import datasets.build_RS_data as build_RS_data

model_saved_path = "sk_svm_trained.pkl"
scaler_saved_path = "scaler_saved.pkl"


def get_output_name(input_tif):
    folder = os.path.dirname(input_tif)
    file_name = os.path.basename(input_tif)
    name_noext = os.path.splitext(file_name)[0]
    return os.path.join(folder, name_noext + "_classified.tif")


def read_training_pixels(image_path, label_path):
    """
    read training pixels from image and the corresponding label
    :param image_path:
    :param label_path:
    :return: X,y array or False
    """

    if io_function.is_file_exist(image_path) is False or io_function.is_file_exist(label_path) is False:
        return False

    # check: they are from the same polygons
    polygon_index_img = os.path.basename(image_path).split('_')[-3]
    polygon_index_label = os.path.basename(label_path).split('_')[-3]
    if polygon_index_img != polygon_index_label:
        raise ValueError("%s and %s are not from the same training polygons" % (image_path, label_path))

    with rasterio.open(image_path) as img_obj:
        # read the all bands
        indexes = img_obj.indexes
        nbands = len(indexes)
        img_data = img_obj.read(indexes)

    with rasterio.open(label_path) as img_obj:
        # read the all bands (only have one band)
        indexes = img_obj.indexes
        if len(indexes) != 1:
            raise ValueError('error, the label should only have one band')

        label_data = img_obj.read(indexes)

    # check the size
    # print(img_data.shape)
    # print(label_data.shape)
    if img_data.shape[1] != label_data.shape[1] or img_data.shape[2] != label_data.shape[2]:
        raise ValueError('the image and label have different size')

    X_arr = img_data.reshape(nbands, -1)
    y_arr = label_data.reshape(-1)

    basic.outputlogMessage(str(X_arr.shape))
    basic.outputlogMessage(str(y_arr.shape))
    # sys.exit(1)

    return X_arr, y_arr


def read_whole_x_pixels(image_path):
    with rasterio.open(image_path) as img_obj:
        # read the all bands
        indexes = img_obj.indexes

        img_data = img_obj.read(indexes)

        nbands, height, width = img_data.shape

        X_arr = img_data.reshape(nbands, -1)
        X_arr = np.transpose(X_arr, (1, 0))
        return X_arr, height, width


class classify_pix_operation(object):
    """perform classify operation on raster images"""

    def __init__(self):
        # Preprocessing
        self.__scaler = None

        # classifiers
        self.__classifier_svm = None
        # self.__classifier_tree = None
        pass

    def __del__(self):
        # release resource
        self.__classifier_svm = None
        # self.__classifier_tree = None
        pass

    def read_training_pixels_from_multi_images(input, subImg_folder, subLabel_folder):
        """
        read pixels from subset images, which are extracted from Planet images based on trainig polygons
        :param subImg_folder: the folder containing images
        :param subLabel_folder: the folder containing labels
        :return: X, y arrays or None
        """
        img_list = io_function.get_file_list_by_ext('.tif', subImg_folder, bsub_folder=False)
        label_list = io_function.get_file_list_by_ext('.tif', subLabel_folder, bsub_folder=False)
        img_list.sort()
        label_list.sort()

        if len(img_list) < 1 or len(label_list) < 1:
            raise IOError('No tif images or labels in folder %s or %s' % (subImg_folder, subLabel_folder))
        if len(img_list) != len(label_list):
            raise ValueError('the number of images is not equal to the one of labels')

        # read them one by one
        Xs, ys = [], []
        for img, label in zip(img_list, label_list):
            # # test by hlc
            # polygon_index_img = os.path.basename(img).split('_')[-3]
            # # print(polygon_index_img)
            # if polygon_index_img not in [str(83), str(86)] :
            #     continue

            X_aImg, y_a = read_training_pixels(img, label)
            Xs.append(X_aImg)
            ys.append(y_a)

        X_pixels = np.concatenate(Xs, axis=1)
        y_pixels = np.concatenate(ys, axis=0)
        X_pixels = np.transpose(X_pixels, (1, 0))
        basic.outputlogMessage(str(X_pixels.shape))
        basic.outputlogMessage(str(y_pixels.shape))

        return X_pixels, y_pixels

    def pre_processing(self, whole_dataset, type=None):
        """
        pre-processing of whole dataset
        :param whole_dataset: the whole dataset
        :param type: pre processing type, such as Standardization, Normalization, Binarization,  Encoding categorical features
        :return:
        """
        # for svm
        X = whole_dataset
        if self.__scaler == None:
            self.__scaler = preprocessing.StandardScaler().fit(X)
        else:
            basic.outputlogMessage('warning, StandardScaler object already exist, this operation will overwrite it')
            self.__scaler = preprocessing.StandardScaler().fit(X)
        # save
        joblib.dump(self.__scaler, scaler_saved_path)

    def training_svm_classifier(self, training_X, training_y):
        """
        train svm classifier
        :param training_data: an array of size [n_records, n_features(fields) + 1 (class) ]
        :return: True if successful, Flase otherwise
        """
        if self.__classifier_svm is None:
            self.__classifier_svm = svm.SVC()  # LinearSVC() #SVC()
        else:
            basic.outputlogMessage('warning, classifier already exist, this operation will replace the old one')
            self.__classifier_svm = svm.SVC()  # LinearSVC()  #SVC()

        if os.path.isfile(scaler_saved_path) and self.__scaler is None:
            self.__scaler = joblib.load(scaler_saved_path)
            result = self.__scaler.transform(training_X)
            X = result.tolist()
        elif self.__scaler is not None:
            result = self.__scaler.transform(training_X)
            X = result.tolist()
        else:
            X = training_X
            basic.outputlogMessage('warning, no pre-processing of data before training')

        y = training_y

        basic.outputlogMessage('Training data set nsample: %d, nfeature: %d' % (len(X), len(X[0])))

        # X_train = X
        # y_train = y
        # # for test by hlc
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.95, random_state=0)
        # X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=0)

        # SVM Parameter Tuning in Scikit Learn using GridSearchCV

        # #Set the parameters by cross-validation
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-4, 0.001, 0.01, 0.1, 1, 2.5, 5],
                             'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
                            {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]

        # # for test by hlc
        # tuned_parameters = [{'kernel': ['linear'], 'C': [0.001,  0.1,1, 10]}]

        clf = model_selection.GridSearchCV(svm.SVC(), tuned_parameters, cv=5,
                                           scoring='f1_macro', n_jobs=-1, verbose=3)

        clf.fit(X_train, y_train)

        basic.outputlogMessage("Best parameters set found on development set:" + str(clf.best_params_))
        basic.outputlogMessage("Grid scores on development set:\n")

        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            basic.outputlogMessage("%0.3f (+/-%0.03f) for %r"
                                   % (mean, std * 2, params))

        # fit_model = self.__classifier_svm.fit(X,y)
        # basic.outputlogMessage(str(fit_model))

        # save the classification model
        joblib.dump(clf, model_saved_path)

    def prediction_on_a_image(self, input, output):
        """
        conduct prediction on a tif image
        :param input:
        :param output:
        :return:
        """

        # load the saved model
        if os.path.isfile(model_saved_path) is False:
            raise IOError('trained model: %s not exist' % model_saved_path)

        clf = joblib.load(model_saved_path)

        # split a large image to many small ones
        patch_w = 2000  # parameters.get_digit_parameters("", "train_patch_width", None, 'int')
        patch_h = 2000  # parameters.get_digit_parameters("", "train_patch_height", None, 'int')
        overlay_x = 0  # parameters.get_digit_parameters("", "train_pixel_overlay_x", None, 'int')
        overlay_y = 0  # parameters.get_digit_parameters("", "train_pixel_overlay_y", None, 'int')

        img_folder = os.path.dirname(input)
        img_name = os.path.basename(input)
        inf_list_txt = 'inf_image_list.txt'
        with open(inf_list_txt, 'w') as txt_obj:
            txt_obj.writelines(img_name + '\n')

        img_patches = build_RS_data.make_dataset(img_folder, inf_list_txt, patch_w, patch_h, overlay_x, overlay_y,
                                                 train=False)

        for p_idx, img_patch in enumerate(img_patches):
            # read images
            patch_data = build_RS_data.read_patch(img_patch)  # read_whole_x_pixels(input)

            nbands, height, width = patch_data.shape

            X_predit = patch_data.reshape(nbands, -1)
            X_predit = np.transpose(X_predit, (1, 0))

            if os.path.isfile(scaler_saved_path) and self.__scaler is None:
                self.__scaler = joblib.load(scaler_saved_path)
                result = self.__scaler.transform(X_predit)
                X = result.tolist()
            elif self.__scaler is not None:
                result = self.__scaler.transform(X_predit)
                X = result.tolist()
            else:
                X = X_predit
                basic.outputlogMessage('warning, no pre-processing of data before prediction')

            # more method on prediction can be foudn in :
            # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
            pre_result = clf.predict(X)

            # save results
            result_img = pre_result.reshape((height, width))

            with rasterio.open(input) as src_obj:
                # Set spatial characteristics of the output object to mirror the input
                kwargs = src_obj.meta
                kwargs.update(
                    dtype=rasterio.uint8,
                    count=1)
                # Create the file
                with rasterio.open(output, 'w', **kwargs) as dst:
                    dst.write_band(1, result_img.astype(rasterio.uint8))
                basic.outputlogMessage("save to %s" % output)

        return True


def main(options, args):
    basic.outputlogMessage('Is_preprocessing:' + str(options.ispreprocess))
    basic.outputlogMessage('Is_training:' + str(options.istraining))

    classify_obj = classify_pix_operation()

    if options.ispreprocess:
        # preprocessing
        input_tif = args[0]
        if os.path.isfile(scaler_saved_path) is False:
            # #read whole data set for pre-processing
            X, _, _ = read_whole_x_pixels(input_tif)
            classify_obj.pre_processing(X)
        else:
            basic.outputlogMessage('warning, scaled model already exist, skip pre-processing')

    elif options.istraining:
        # training
        # read training data (make sure 'subImages', 'subLabels' is under current folder)
        X, y = classify_obj.read_training_pixels_from_multi_images('subImages', 'subLabels')

        if os.path.isfile(model_saved_path) is False:
            classify_obj.training_svm_classifier(X, y)
        else:
            basic.outputlogMessage("warning, trained model already exist, skip training")

    else:
        # prediction
        input_tif = args[0]
        if options.output is not None:
            output = options.output
        else:
            output = get_output_name(input_tif)
        basic.outputlogMessage('staring prediction on image:' + str(input_tif))
        classify_obj.prediction_on_a_image(input_tif, output)


if __name__ == "__main__":
    usage = "usage: %prog [options] input_image"
    parser = OptionParser(usage=usage, version="1.0 2019-1-4")
    parser.description = 'Introduction: Using SVM in sklearn library to perform classification on Planet images'

    parser.add_option("-p", "--ispreprocess",
                      action="store_true", dest="ispreprocess", default=False,
                      help="to indicate the script will perform pre-processing, if this set, istraining will be ignored")

    parser.add_option("-t", "--istraining",
                      action="store_true", dest="istraining", default=False,
                      help="to indicate the script will perform training process")

    parser.add_option("-o", "--output",
                      action="store", dest="output",
                      help="the output file path")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    basic.setlogfile('planet_svm_log.txt')

    main(options, args)
