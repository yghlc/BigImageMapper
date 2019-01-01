#!/usr/bin/env python
# Filename: split_image
"""
introduction: assess the classification results of remote sensing images

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 22 March, 2018
"""

import sys,os,subprocess
from optparse import OptionParser

import numpy as np
from sklearn import metrics
import rasterio

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

        data_1d = data.flatten()

        return data_1d


def main(options, args):

    label_image = args[0]
    classified_results = args[1]

    no_data = options.no_data

    print("Classification assessment")
    print("classified_result: " +classified_results)
    print("ground true: "+label_image)

    #read image
    label_1d = read_oneband_image_to_1dArray(label_image)
    classified_results_1d = read_oneband_image_to_1dArray(classified_results)

    if label_1d is None or classified_results_1d is None:
        return

    #ignore background (no_data area)
    back_ground_index = np.where(label_1d==no_data)
    # back_ground_index = np.where(classified_results_1d == 0)
    label_1d = np.delete(label_1d,back_ground_index)
    classified_results_1d = np.delete(classified_results_1d, back_ground_index)

    accuracy = metrics.accuracy_score(label_1d,classified_results_1d,normalize=True)*100

    print("accuracy: %.2f %%"%accuracy)

    confusion_matrix = metrics.confusion_matrix(label_1d, classified_results_1d)
    #print("confusion matrix:")
    #print(str(confusion_matrix))

    sum_trace = np.trace(confusion_matrix)
    sum_all = np.sum(confusion_matrix)
    overall_accuracy = sum_trace*100.0/sum_all
    print("overall accuracy: %.2f %%" % overall_accuracy)



    # report = metrics.classification_report(label_1d,classified_results_1d)
    # print(report)
    # print("precision (tp/(tp+fp)) score: %.4f %%" % precision)
    # print("f1 score: %.4f %%" % f1)





if __name__ == "__main__":
    usage = "usage: %prog [options] label_image classified_result"
    parser = OptionParser(usage=usage, version="1.0 2018-3-22")
    parser.description = 'Introduction: assess the classification results of remote sensing images '

    parser.add_option('-n','--no_data',
                      action='store',dest='no_data',type=int,default=255,
                      help="the folder of split image patches ")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    # if options.para_file is None:
    #     basic.outputlogMessage('error, parameter file is required')
    #     sys.exit(2)

    main(options, args)

