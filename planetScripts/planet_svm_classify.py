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

# pip install imbalanced-learn for sub sample the training data.
import imblearn

# Preprocessing
from sklearn import preprocessing
# library for SVM classifier
from sklearn import svm

# model_selection  # change grid_search to model_selection
from sklearn import model_selection

from sklearn.externals import joblib  # save and load model

import datasets.build_RS_data as build_RS_data

import multiprocessing
from multiprocessing import Pool

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

def inference_one_patch_svm(img_idx,image_count,p_idx,patch_count,inf_output_dir,img_patch,scaler,clf):
    """
    inference one patch
    :param img_idx: index of the image
    :param idx: index of the patch on the image
    :param org_img_path: org image path
    :param boundary: the patch boundary
    :param model: sk-learn, svm model
    :return:
    """
    # due to multiprocessing:  the Pickle.PicklingError: Can't pickle <type 'module'>: attribute lookup __builtin__.module failed
    # recreate the class instance, but there is a model from tensorflow, so it sill not work

    # read images
    patch_data = build_RS_data.read_patch(img_patch)  # read_whole_x_pixels(input)

    nbands, height, width = patch_data.shape

    X_predit = patch_data.reshape(nbands, -1)
    X_predit = np.transpose(X_predit, (1, 0))

    if os.path.isfile(scaler_saved_path) and scaler is None:
        scaler = joblib.load(scaler_saved_path)
        result = scaler.transform(X_predit)
        X = result.tolist()
    elif scaler is not None:
        result = scaler.transform(X_predit)
        X = result.tolist()
    else:
        X = X_predit
        basic.outputlogMessage('warning, no pre-processing of data before prediction')

    # more method on prediction can be foudn in :
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    pre_result = clf.predict(X)
    result_img = pre_result.reshape((height, width))

    # save results
    print('Save patch:%d/%d on Image:%d/%d , shape:(%d,%d)' %
          (p_idx,patch_count,img_idx, image_count, result_img.shape[0], result_img.shape[1]))

    # short the file name to avoid  error of " Argument list too long", hlc 2018-Oct-29
    file_name = "I%d_%d" % (img_idx, p_idx)

    save_path = os.path.join(inf_output_dir, file_name + '.tif')
    build_RS_data.save_patch_oneband_8bit(img_patch,result_img.astype(np.uint8),save_path)

    return True


class classify_pix_operation(object):
    """perform classify operation on raster images"""

    def __init__(self):
        # Preprocessing
        self._scaler = None

        # classifiers
        self._classifier = None
        # self._classifier_tree = None
        pass

    def __del__(self):
        # release resource
        self._classifier = None
        # self._classifier_tree = None
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

    def read_training_pixels_inside_polygons(self, img_path, shp_path):
        '''
        read pixels on a image in the extent of polygons
        :param img_path: the path of an image
        :param shp_path: the path of shape file
        :return:
        '''
        if io_function.is_file_exist(img_path) is False or io_function.is_file_exist(shp_path) is False:
            return False

        no_data = 255   # consider changing to other values
        touch = False   # we only read the pixels inside the polygons, so set all_touched as False
        sub_images, class_labels = build_RS_data.read_pixels_inside_polygons(img_path,shp_path,mask_no_data=no_data, touch=touch)

        # read them one by one
        Xs, ys = [], []
        for img_data, label in zip(sub_images, class_labels):
            # img: 3d array (nband, height, width)
             # label: int values

            # print(img_data)
            # print(label)
            X_arr = img_data.reshape(img_data.shape[0], -1)

            # remove non-data pixels
            valid_pixles = np.any(X_arr != no_data,axis=0)
            X_arr = X_arr[:,valid_pixles]

            y_arr = np.ones(X_arr.shape[1])*label
            Xs.append(X_arr)
            ys.append(y_arr)

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
        if self._scaler == None:
            self._scaler = preprocessing.StandardScaler().fit(X)
        else:
            basic.outputlogMessage('warning, StandardScaler object already exist, this operation will overwrite it')
            self._scaler = preprocessing.StandardScaler().fit(X)
        # save
        joblib.dump(self._scaler, scaler_saved_path)

    def training_svm_classifier(self, training_X, training_y):
        """
        train svm classifier
        :param training_data: an array of size [n_records, n_features(fields) + 1 (class) ]
        :return: True if successful, Flase otherwise
        """
        if training_X is None or training_y is None:
            raise ValueError('the training samples are None')

        if self._classifier is None:
            self._classifier = svm.SVC()  # LinearSVC() #SVC()
        else:
            basic.outputlogMessage('warning, classifier already exist, this operation will replace the old one')
            self._classifier = svm.SVC()  # LinearSVC()  #SVC()

        if os.path.isfile(scaler_saved_path) and self._scaler is None:
            self._scaler = joblib.load(scaler_saved_path)
            result = self._scaler.transform(training_X)
            X = result.tolist()
        elif self._scaler is not None:
            result = self._scaler.transform(training_X)
            X = result.tolist()
        else:
            X = training_X
            basic.outputlogMessage('warning, no pre-processing of data before training')

        y = training_y

        basic.outputlogMessage('Training data set nsample: %d, nfeature: %d' % (len(X), len(X[0])))

        # # sub sample and make the class 0 and 1 balanced (have the same number)
        # basic.outputlogMessage('Number of sample before sub-sample: %d, class 0: %d, class 1: %d'%
        #                        (len(X),len(np.where(y==0)[0]),len(np.where(y==1)[0])))
        # from imblearn.under_sampling import RandomUnderSampler
        # rus = RandomUnderSampler(return_indices=True)
        # X_rus, y_rus, id_rus = rus.fit_sample(X, y)
        # X = X_rus
        # y = y_rus
        # basic.outputlogMessage('Number of sample after sub-sample: %d, class 0: %d, class 1: %d'%
        #                        (len(X),len(np.where(y==0)[0]),len(np.where(y==1)[0])))

        X_train = X
        y_train = y
        # # for test by hlc
        # X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.95, random_state=0)
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

        # fit_model = self._classifier.fit(X,y)
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
        patch_w = 500  # parameters.get_digit_parameters("", "train_patch_width", None, 'int')
        patch_h = 500  # parameters.get_digit_parameters("", "train_patch_height", None, 'int')
        overlay_x = 0  # parameters.get_digit_parameters("", "train_pixel_overlay_x", None, 'int')
        overlay_y = 0  # parameters.get_digit_parameters("", "train_pixel_overlay_y", None, 'int')

        img_folder = os.path.dirname(input)
        img_name = os.path.basename(input)
        inf_list_txt = 'inf_image_list.txt'
        with open(inf_list_txt, 'w') as txt_obj:
            txt_obj.writelines(img_name + '\n')

        img_patches = build_RS_data.make_dataset(img_folder, inf_list_txt, patch_w, patch_h, overlay_x, overlay_y,
                                                 train=False)

        for img_idx, aImg_patches in enumerate(img_patches):
            inf_output_dir = 'inf_results' #os.path.splitext(img_name)[0]
            os.system('mkdir -p '+inf_output_dir)
            os.system('rm '+inf_output_dir+'/*')

            ## parallel inference patches
            # but it turns out not work due to the Pickle.PicklingError
            # not working due to mulitple parameters. Jan 9, 2019, hlc
            # use multiple thread
            num_cores = multiprocessing.cpu_count()
            print('number of thread %d' % num_cores)
            # theadPool = mp.Pool(num_cores)  # multi threads, can not utilize all the CPUs? not sure hlc 2018-4-19
            theadPool = Pool(num_cores)  # multi processes

            # inference_one_patch_svm(img_idx, image_count, p_idx, patch_count, inf_output_dir, img_patch, scaler,clf)

            parameters_list = [
                (img_idx, len(img_patches), idx, len(aImg_patches), inf_output_dir, img_patch, self._scaler, clf)
                for (idx, img_patch) in enumerate(aImg_patches)]
            # results = theadPool.map(inference_one_patch_svm, parameters_list)     # not working
            results = theadPool.starmap(inference_one_patch_svm, parameters_list)   # need python3
            print('result_list', results)

            # for p_idx, img_patch in enumerate(aImg_patches):
            #     # read images
            #     patch_data = build_RS_data.read_patch(img_patch)  # read_whole_x_pixels(input)
            #
            #     nbands, height, width = patch_data.shape
            #
            #     X_predit = patch_data.reshape(nbands, -1)
            #     X_predit = np.transpose(X_predit, (1, 0))
            #
            #     if os.path.isfile(scaler_saved_path) and self._scaler is None:
            #         self._scaler = joblib.load(scaler_saved_path)
            #         result = self._scaler.transform(X_predit)
            #         X = result.tolist()
            #     elif self._scaler is not None:
            #         result = self._scaler.transform(X_predit)
            #         X = result.tolist()
            #     else:
            #         X = X_predit
            #         basic.outputlogMessage('warning, no pre-processing of data before prediction')
            #
            #     # more method on prediction can be foudn in :
            #     # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
            #     pre_result = clf.predict(X)
            #     result_img = pre_result.reshape((height, width))
            #
            #     # save results
            #     print('Save patch:%d/%d on Image:%d/%d , shape:(%d,%d)' %
            #           (p_idx,len(aImg_patches), img_idx,len(img_patches), result_img.shape[0], result_img.shape[1]))
            #
            #     # short the file name to avoid  error of " Argument list too long", hlc 2018-Oct-29
            #     file_name = "I%d_%d" % (img_idx, p_idx)
            #
            #     save_path = os.path.join(inf_output_dir, file_name + '.tif')
            #     build_RS_data.save_patch_oneband_8bit(img_patch,result_img.astype(np.uint8),save_path)
            #
            #     with rasterio.open(input) as src_obj:
            #         # Set spatial characteristics of the output object to mirror the input
            #         kwargs = src_obj.meta
            #         kwargs.update(
            #             dtype=rasterio.uint8,
            #             count=1)
            #         # Create the file
            #         with rasterio.open(output, 'w', **kwargs) as dst:
            #             dst.write_band(1, result_img.astype(rasterio.uint8))
            #         basic.outputlogMessage("save to %s" % output)

        return True


def main(options, args):
    basic.outputlogMessage('Is_preprocessing:' + str(options.ispreprocess))
    basic.outputlogMessage('Is_training:' + str(options.istraining))

    classify_obj = classify_pix_operation()
    input_tif = args[0]

    if options.ispreprocess:
        # preprocessing
        if os.path.isfile(scaler_saved_path) is False:
            # #read whole data set for pre-processing
            X, _, _ = read_whole_x_pixels(input_tif)
            classify_obj.pre_processing(X)
        else:
            basic.outputlogMessage('warning, scaled model already exist, skip pre-processing')

    elif options.istraining:
        # training
        if options.polygon_train is None:
        # read training data (make sure 'subImages', 'subLabels' is under current folder)
            X, y = classify_obj.read_training_pixels_from_multi_images('subImages', 'subLabels')
        else:
            X, y = classify_obj.read_training_pixels_inside_polygons(input_tif, options.polygon_train)

        if os.path.isfile(model_saved_path) is False:
            classify_obj.training_svm_classifier(X, y)
        else:
            basic.outputlogMessage("warning, trained model already exist, skip training")

    else:
        # prediction
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

    parser.add_option("-s", "--shape_train",
                      action="store", dest="polygon_train",
                      help="the shape file containing polygons for training")

    parser.add_option("-o", "--output",
                      action="store", dest="output",
                      help="the output file path")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    basic.setlogfile('planet_svm_log.txt')

    main(options, args)
