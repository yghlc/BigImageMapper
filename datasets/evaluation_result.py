#!/usr/bin/env python
# Filename: evaluation_result 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 13 December, 2019
"""

from optparse import OptionParser
import os,sys
import basic_src.basic as basic
import basic_src.io_function as io_function
import parameters

import vector_features
from vector_features import shape_opeation

def evaluation_result(result_shp,val_shp,evaluation_txt=None):
    """
    evaluate the result based on IoU
    :param result_shp: result shape file contains detected polygons
    :param val_shp: shape file contains validation polygons
    :return: True is successful, False otherwise
    """
    basic.outputlogMessage("evaluation result")
    IoUs = vector_features.calculate_IoU_scores(result_shp,val_shp)
    if IoUs is False:
        return False

    #save IoU to result shapefile
    operation_obj = shape_opeation()
    operation_obj.add_one_field_records_to_shapefile(result_shp, IoUs, 'IoU')

    iou_threshold = parameters.get_IOU_threshold()
    true_pos_count = 0
    false_pos_count = 0
    val_polygon_count = operation_obj.get_shapes_count(val_shp)
    # calculate precision, recall, F1 score
    for iou in IoUs:
        if iou > iou_threshold:
            true_pos_count  +=  1
        else:
            false_pos_count += 1

    false_neg_count = val_polygon_count - true_pos_count
    if false_neg_count < 0:
        basic.outputlogMessage('warning, false negative count is smaller than 0, recall can not be trusted')

    precision = float(true_pos_count) / (float(true_pos_count) + float(false_pos_count))
    recall = float(true_pos_count) / (float(true_pos_count) + float(false_neg_count))
    if (true_pos_count > 0):
        F1score = 2.0 * precision * recall / (precision + recall)
    else:
        F1score = 0

    #output evaluation reslult
    if evaluation_txt is None:
        evaluation_txt = "evaluation_report.txt"
    f_obj = open(evaluation_txt,'w')
    f_obj.writelines('true_pos_count: %d\n'%true_pos_count)
    f_obj.writelines('false_pos_count: %d\n'% false_pos_count)
    f_obj.writelines('false_neg_count: %d\n'%false_neg_count)
    f_obj.writelines('precision: %.6f\n'%precision)
    f_obj.writelines('recall: %.6f\n'%recall)
    f_obj.writelines('F1score: %.6f\n'%F1score)
    f_obj.close()

    ##########################################################################################
    ## another method for calculating false_neg_count base on IoU value
    # calculate the IoU for validation polygons (ground truths)
    IoUs = vector_features.calculate_IoU_scores(val_shp, result_shp)
    if IoUs is False:
        return False

    # if the IoU of a validation polygon smaller than threshold, then it's false negative
    false_neg_count = 0
    idx_of_false_neg = []
    for idx,iou in enumerate(IoUs):
        if iou < iou_threshold:
            false_neg_count +=  1
            idx_of_false_neg.append(idx+1) # index start from 1

    precision = float(true_pos_count) / (float(true_pos_count) + float(false_pos_count))
    recall = float(true_pos_count) / (float(true_pos_count) + float(false_neg_count))
    if (true_pos_count > 0):
        F1score = 2.0 * precision * recall / (precision + recall)
    else:
        F1score = 0
    # output evaluation reslult

    # evaluation_txt = "evaluation_report.txt"
    f_obj = open(evaluation_txt, 'a')  # add to "evaluation_report.txt"
    f_obj.writelines('\n\n** Count false negative by IoU**\n')
    f_obj.writelines('true_pos_count: %d\n' % true_pos_count)
    f_obj.writelines('false_pos_count: %d\n' % false_pos_count)
    f_obj.writelines('false_neg_count_byIoU: %d\n' % false_neg_count)
    f_obj.writelines('precision: %.6f\n' % precision)
    f_obj.writelines('recall: %.6f\n' % recall)
    f_obj.writelines('F1score: %.6f\n' % F1score)
    # output the index of false negative
    f_obj.writelines('\nindex (start from 1) of false negatives: %s\n' % ','.join([str(item) for item in idx_of_false_neg]))
    f_obj.close()

    pass

def main(options, args):
    input = args[0]

    # evaluation result
    multi_val_files = parameters.get_string_parameters_None_if_absence('','validation_shape_list')
    if multi_val_files is None:
        val_path = parameters.get_validation_shape()
    else:
        val_path = io_function.get_path_from_txt_list_index(multi_val_files)
        # try to change the home folder path if the file does not exist
        val_path = io_function.get_file_path_new_home_folder(val_path)

    if os.path.isfile(val_path):
        basic.outputlogMessage('Start evaluation, input: %s, validation file: %s'%(input, val_path))
        evaluation_result(input, val_path)
    else:
        basic.outputlogMessage("warning, validation polygon (%s) not exist, skip evaluation"%val_path)



if __name__=='__main__':
    usage = "usage: %prog [options] input_path "
    parser = OptionParser(usage=usage, version="1.0 2017-7-24")
    parser.description = 'Introduction: evaluate the mapping results '
    parser.add_option("-p", "--para",
                      action="store", dest="para_file",
                      help="the parameters file")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)
    ## set parameters files
    if options.para_file is None:
        print('error, no parameters file')
        parser.print_help()
        sys.exit(2)
    else:
        parameters.set_saved_parafile_path(options.para_file)

    main(options, args)
