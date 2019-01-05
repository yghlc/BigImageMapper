#!/usr/bin/env python
# Filename: planet_svm_classify
"""
introduction: Using SVM in sklearn library to perform classification on Planet images

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 4 January, 2019
"""


import sys,os
from optparse import OptionParser

import rasterio
import numpy as np

HOME = os.path.expanduser('~')
# path of DeeplabforRS
codes_dir2 = HOME +'/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)

import basic_src.io_function as io_function
import basic_src.basic as basic

# Preprocessing
from sklearn import preprocessing
# library for SVM classifier
from sklearn import svm

#model_selection  # change grid_search to model_selection
from sklearn import model_selection


def get_output_name(input_tif):
    folder = os.path.dirname(input_tif)
    file_name = os.path.basename(input_tif)
    name_noext = os.path.splitext(file_name)[0]
    return os.path.join(folder, name_noext + "_classified.tif")

def read_training_pixels(image_path,label_path):
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
        raise ValueError("%s and %s are not from the same training polygons"%(image_path,label_path))

    with rasterio.open(image_path) as img_obj:
        # read the all bands
        indexes = img_obj.indexes
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
    if img_data.shape[1]!=label_data.shape[1] or img_data.shape[2]!=label_data.shape[2]:
        raise ValueError('the image and label have different size')

    X_arr = img_data.reshape(3,-1)
    y_arr = label_data.reshape(-1)

    print(X_arr.shape)
    print(y_arr.shape)
    # sys.exit(1)

    return X_arr,y_arr


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

    def read_training_pixels_from_multi_images(input, subImg_folder,subLabel_folder):
        """
        read pixels from subset images, which are extracted from Planet images based on trainig polygons
        :param subImg_folder: the folder containing images
        :param subLabel_folder: the folder containing labels
        :return: X, y arrays or None
        """
        img_list = io_function.get_file_list_by_ext('.tif',subImg_folder,bsub_folder=False)
        label_list = io_function.get_file_list_by_ext('.tif',subLabel_folder,bsub_folder=False)
        img_list.sort()
        label_list.sort()

        if len(img_list)<1 or len(label_list) < 1:
            raise IOError('No tif images or labels in folder %s or %s'%(subImg_folder,subLabel_folder))
        if len(img_list) != len(label_list):
            raise ValueError('the number of images is not equal to the one of labels')

        # read them one by one
        Xs, ys = [],[]
        for img, label in zip(img_list,label_list):

            # test
            polygon_index_img = os.path.basename(img).split('_')[-3]
            # print(polygon_index_img)
            if polygon_index_img not in [str(83), str(86)] :
                continue

            X_aImg,y_a = read_training_pixels(img, label)
            Xs.append(X_aImg)
            ys.append(y_a)

        X_pixels = np.concatenate(Xs,axis=1)
        y_pixels = np.concatenate(ys,axis=0)
        X_pixels = np.transpose(X_pixels,(1,0))
        print(X_pixels.shape)
        print(y_pixels.shape)

        return X_pixels, y_pixels

    def pre_processing(self,whole_dataset,type=None):
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

    def training_svm_classifier(self,training_X, training_y):
        """
        train svm classifier
        :param training_data: an array of size [n_records, n_features(fields) + 1 (class) ]
        :return: True if successful, Flase otherwise
        """
        if self.__classifier_svm is None:
            self.__classifier_svm = svm.SVC() #LinearSVC() #SVC()
        else:
            basic.outputlogMessage('warning, classifier already exist, this operation will replace the old one')
            self.__classifier_svm = svm.SVC() #LinearSVC()  #SVC()

        if self.__scaler is not None:
            result = self.__scaler.transform(training_X)
            X  = result.tolist()
        y = training_y

        basic.outputlogMessage('Training data set nsample: %d, nfeature: %d' % (len(X), len(X[0])))

        # X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=0)

        # SVM Parameter Tuning in Scikit Learn using GridSearchCV

        # Set the parameters by cross-validation
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-4,0.001, 0.01, 0.1, 1,2.5, 5],
                             'C': [0.001, 0.01, 0.1,1, 10, 100, 1000]},
                            {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1,1, 10, 100, 1000]}]

        clf = model_selection.GridSearchCV(svm.SVC(), tuned_parameters, cv=5,
                           scoring='f1' ,n_jobs=-1)

        clf.fit(X, y)

        basic.outputlogMessage("Best parameters set found on development set:"+str(clf.best_params_))
        basic.outputlogMessage("Grid scores on development set:" )
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        # print()

        # fit_model = self.__classifier_svm.fit(X,y)
        # basic.outputlogMessage(str(fit_model))


def main(options, args):
    input_tif = args[0]
    label_img = args[1]

    if options.output is not None:
        output = options.output
    else:
        output = get_output_name(input_tif)


    classify_obj = classify_pix_operation()
    #
    # #read whole data set for pre-processing
    X, y = classify_obj.read_training_pixels_from_multi_images('subImages','subLabels')
    classify_obj.pre_processing(X)
    #
    # read training data
    # (the same as the previous one )

    classify_obj.training_svm_classifier(X,y)

    #save train model
    #
    # # classify
    # classify_obj.classify_polygon_svm(shapefile, output_shp)

    # classifier_obj = None



    pass



if __name__ == "__main__":
    usage = "usage: %prog [options] input_image label_image"
    parser = OptionParser(usage=usage, version="1.0 2019-1-4")
    parser.description = 'Introduction: Using SVM in sklearn library to perform classification on Planet images'

    parser.add_option("-o", "--output",
                      action="store", dest="output",
                      help="the output file path")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)