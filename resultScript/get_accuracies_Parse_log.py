#!/usr/bin/env python
# Filename:
"""
introduction: parse the log file of "plot_accuracies.py", i.e., accuracies_log.txt
# then output the accuracies to a table

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 12 February, 2019
"""
import os,sys
import re

from optparse import OptionParser

import pandas as pd

# only output the accuracies when IOU_thr are in
# output_iou=[0.8,0.6,0.4,0.2,0]
output_iou=[0,0.4,0.8]
result_list = []

def read_txt_file(path):
    if os.path.isfile(path) is False:
        raise IOError('%s not exist'%path)
    with open(path) as f_obj:
        lines_str = f_obj.readlines()
        return lines_str

def get_iou(line_str):
    '''
    get iou value from the string
    :param line_str:
    :return:
    '''
    return float(re.findall('\d+\.\d+',line_str)[0])

def get_tp_fp_fn_str(line_str):
    '''
    parse the tp, fp, fn from on string
    :param line_str:
    :return: string of the number
    '''
    # iou_thr = float(re.findall('\d+\.\d+',line_str)[0])
    integer = re.findall('\d+',line_str)
    tp = integer[-5]
    fp = integer[-4]
    fn = integer[-3]
    return tp , fp, fn

def get_precision_recall_f1score_str(line_str):
    '''
    parse the precision, recall, f1score from on string
    :param line_str:
    :return: string of the number
    '''
    float_str = re.findall('\d+\.\d+',line_str)
    precision = float_str[1]
    recall = float_str[2]
    f1score = float_str[3]
    return precision, recall, f1score

def get_test_num(file_name):
    '''
    get test num from the shapefile name
    :param line_str:
    :return:
    '''
    # if 'fold' in file_name:  # k-fold cross-validation
    #     # tmp = file_name.split('_')
    #     # label = '_'.join(tmp[-3:])
    #     raise ValueError('unknow test type')
    if 'imgAug' in file_name:  # image augmentation test
        fn_no_ext = os.path.splitext(file_name)[0]
        tmp = fn_no_ext.split('_')
        label = tmp[-1]
        test_num = re.findall('\d+',label)[0]
        return test_num
    else:
        print('it is not test on data agumentation, skip get_get_num')
        return False
        # raise ValueError('unknow test type')

def check_duplicated_records(shapefile_str):
    for res in result_list:
        if res['shapefile'] == shapefile_str:
            print("Warning, %s in the acc_log file has duplicates, skip it")
            return True
    return False

def parse_acc_log_file(acc_log_file):
    '''

    :param acc_log_file:
    :return:
    '''

    # parse the logfile
    log_lines = read_txt_file(acc_log_file)
    l_idx=0 # only allow to increase`
    line_count = len(log_lines)

    while(l_idx<line_count):
        result = {}

        if "calculate precision recall curve for" in log_lines[l_idx]:
            tmp_str = log_lines[l_idx].split(' for ')[-1].strip()
            # print(tmp_str)
            shapefile_str = os.path.basename(tmp_str)
            if check_duplicated_records(shapefile_str):
                continue
            else:
                result['shapefile']  = shapefile_str
            l_idx += 1

            iou_thr_list = []
            TP_list = []
            FP_list = []
            FN_list = []

            precision_list = []
            recall_list = []
            f1score_list = []

            # get the accuracies
            while 'iou_thr' in log_lines[l_idx]:
                iou_value = get_iou(log_lines[l_idx])
                if iou_value not in output_iou:
                    l_idx += 2  # skip two lines
                    continue
                iou_thr_list.append(str(iou_value))
                # get TP, FP, FN
                TP, FP, FN = get_tp_fp_fn_str(log_lines[l_idx])
                TP_list.append(TP)
                FP_list.append(FP)
                FN_list.append(FN)

                # get precision, recall, f1_score
                l_idx += 1
                precision, recall, f1_score = get_precision_recall_f1score_str(log_lines[l_idx])
                precision_list.append(precision)
                recall_list.append(recall)
                f1score_list.append(f1_score)

                # move to next line
                l_idx += 1

            result['iou_thr_list'] = iou_thr_list
            result['tp_list'] = TP_list
            result['fp_list'] = FP_list
            result['fn_list'] = FN_list
            result['precision_list'] = precision_list
            result['recall_list'] = recall_list
            result['f1score_list'] = f1score_list

        else:
            # move to next line
            l_idx += 1
            continue

        result_list.append(result)


def parse_average_precision_file(average_prec):

    # parse the p_r_img_augmentation_ap.txt
    ap_lines = read_txt_file(average_prec)
    ap_lines = ap_lines[1:]    # remove the first line
    for ap_line in ap_lines:
        # if this is headling line, skip it (heading exists in the middle of files when the ap file was merged from multiple files)
        if 'average_precision' in ap_line:
            continue
        ap_str = re.findall('\d+\.\d+',ap_line)[0]
        shapefile = os.path.basename(ap_line.split()[0])

        for result in result_list:
            if result['shapefile'] == shapefile:
                result['average_precision'] = ap_str
                break

def parse_time_cost(time_cost_file):
    '''

    :param time_cost_file:
    :return:
    '''
    # parse the time_cost.txt
    if os.path.isfile(time_cost_file) is False:
        return False
    time_lines = read_txt_file(time_cost_file)
    time_lines = [item.strip() for item in time_lines]
    for result in result_list:
        test_num = get_test_num(result['shapefile'])
        if test_num is False:
            return False
        result['test_num'] = int(test_num)  # add test number for reorder
        test_num_str = 'test_num:'+test_num
        if test_num_str in time_lines:
            pass
            idx_list = [i for i, line in enumerate(time_lines) if line == test_num_str]
            # print(idx_list)
            for idx in idx_list:
                if 'count of class 1' in time_lines[idx+2]: # check the count exist
                    result['class_1_count'] = time_lines[idx+2].split(':')[1]
                    result['class_0_count'] = time_lines[idx + 1].split(':')[1]
                    result['img_aug_str'] = time_lines[idx - 1].split(':')[1]
                    break
        else:
            print('warning: test_num:%s not in the file'%test_num)

def reorder_result():
    '''
    rearrange the order of result based on test number
    :return:
    '''

    if 'test_num' not in result_list[0].keys():
        print('dont have test number, skip reording')
        return False
    # sorted(result_list, key=lambda x: result_list[x]['test_num'])
    # for result in result_list:
    #     print(result['test_num'])
    result_list.sort(key=lambda x: x['test_num'])


def save_to_csv_file(save_path):
    # save to file
    import csv
    with open(save_path,'w') as f_obj:
        for result in result_list:
            for key in result.keys():
                f_obj.write("%s,%s\n" % (key, result[key]))

    csv_columns = ['shapefile','average_precision','img_aug_str','class_0_count','class_1_count',
                   'iou_thr','TruePositve','FalsePositive','FalseNegative','precision','recall','F1score']
    # to 2D table
    dict_data = []
    for result in result_list:
        for idx,iou_thr in enumerate(result['iou_thr_list']):
            record ={}
            record['shapefile'] = result['shapefile']
            print(result['shapefile'])
            if 'average_precision' in result.keys(): record['average_precision'] = result['average_precision']
            else: record['average_precision'] = ''
            if 'img_aug_str' in result.keys():  record['img_aug_str']= result['img_aug_str']
            else: record['img_aug_str'] = ''
            if 'class_0_count' in result.keys():  record['class_0_count']= result['class_0_count']
            else: record['class_0_count'] = ''
            if 'class_1_count' in result.keys():  record['class_1_count']= result['class_1_count']
            else: record['class_1_count'] = ''

            record['iou_thr'] = iou_thr
            record['TruePositve'] = result['tp_list'][idx]
            record['FalsePositive'] = result['fp_list'][idx]
            record['FalseNegative'] = result['fn_list'][idx]
            record['precision'] = result['precision_list'][idx]
            record['recall'] = result['recall_list'][idx]
            record['F1score'] = result['f1score_list'][idx]
            dict_data.append(record)

    try:
        with open(save_path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error of %s"%save_path)

def main(options, args):

    # acc_log_file = '/home/hlc/Dropbox/a_canada_sync/accuracies_log.txt'
    # average_prec = '/home/hlc/Dropbox/a_canada_sync/p_r_img_augmentation_ap.txt'
    # time_cost_file = '/home/hlc/Dropbox/a_canada_sync/time_cost.txt'

    acc_log_file = args[0]
    average_prec =  args[1]
    if len(args)>2:
        time_cost_file = args[2]
    else:
        time_cost_file = ''

    parse_acc_log_file(acc_log_file)

    parse_average_precision_file(average_prec)

    parse_time_cost(time_cost_file)

    reorder_result()

    csv_file = options.output # "accuracy_table.csv"
    save_to_csv_file(csv_file)

    # save a copy to xlsx file
    if os.path.isfile(csv_file):
        xlsx_file = os.path.splitext(csv_file)[0] + '.xlsx'
        read_file = pd.read_csv(csv_file)
        read_file.to_excel(xlsx_file, index=None, header=True)
        print("save a copy of csv file (%s) to excel file (%s)"%(csv_file,xlsx_file))


if __name__ == "__main__":
    usage = "usage: %prog [options] acc_log_file average_prec time_cost_file"
    parser = OptionParser(usage=usage, version="1.0 2019-1-4")
    parser.description = 'Introduction: parse the log file and save to a table'

    parser.add_option("-o", "--output",
                      action="store", dest="output",default="accuracy_table.csv",
                      help="the output file path")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)
