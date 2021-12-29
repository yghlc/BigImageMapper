#!/usr/bin/env python
# Filename: eva_report_to_tables 
"""
introduction: convert multiple evaluation reports to tables.

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 28 June, 2020
"""
import os,sys
import re

from optparse import OptionParser

import pandas as pd

def get_tp_fp_fn_etc(six_lines):
    tp = int(six_lines[0].split(':')[1].strip())
    fp = int(six_lines[1].split(':')[1].strip())
    fn = int(six_lines[2].split(':')[1].strip())
    precision = float(six_lines[3].split(':')[1].strip())
    recall = float(six_lines[4].split(':')[1].strip())
    F1score = float(six_lines[5].split(':')[1].strip())
    return [tp, fp, fn, precision, recall, F1score]

def read_eva_report(file_path, count_iou_version=False):
    with open(file_path) as fp:
        lines = fp.readlines()
        line_count = len(lines)
        if count_iou_version is True:
            for idx in range(line_count):
                if 'IoU' in lines[idx]:
                    return get_tp_fp_fn_etc(lines[idx+1:idx+7])

        return get_tp_fp_fn_etc(lines[:6])

def read_accuracy_multi_reports(eva_reports):
    print('Input %d reports:'%len(eva_reports))
    for report in eva_reports:
        print(report)

    file_path_list = []
    file_name_list = []
    TP_list = []
    FP_list = []
    FN_list = []
    precision_list = []
    recall_list = []
    F1_score_list = []

    TP_iou_version_list = []
    FP_iou_version_list = []
    FN_iou_version_list = []
    precision_iou_version_list = []
    recall_iou_version_list = []
    F1_score_iou_version_list = []


    for report in eva_reports:
        acc_etc = read_eva_report(report)
        acc_etc_iou_ver = read_eva_report(report,count_iou_version=True)
        file_path_list.append(report)
        file_name_list.append(os.path.basename(report))

        TP_list.append(acc_etc[0])
        FP_list.append(acc_etc[1])
        FN_list.append(acc_etc[2])
        precision_list.append(acc_etc[3])
        recall_list.append(acc_etc[4])
        F1_score_list.append(acc_etc[5])

        TP_iou_version_list.append(acc_etc_iou_ver[0])
        FP_iou_version_list.append(acc_etc_iou_ver[1])
        FN_iou_version_list.append(acc_etc_iou_ver[2])
        precision_iou_version_list.append(acc_etc_iou_ver[3])
        recall_iou_version_list.append(acc_etc_iou_ver[4])
        F1_score_iou_version_list.append(acc_etc_iou_ver[5])

    acc_table = {'file_path':file_path_list, 'file_name':file_name_list, 'TP':TP_list,
                 'FP':FP_list, 'FN':FN_list,'precision':precision_list, 'recall':recall_list,'F1score':F1_score_list}

    acc_table_IOU_version = {'file_path':file_path_list, 'file_name':file_name_list,'TP':TP_iou_version_list,'FP':FP_iou_version_list,
                             'FN':FN_iou_version_list,'precision':precision_iou_version_list,
                             'recall':recall_iou_version_list,'F1score':F1_score_iou_version_list}
    return acc_table, acc_table_IOU_version

def eva_reports_to_table(eva_reports, output_file):

    eva_reports.sort()

    eva_report_rmTimeiou = [item for item in eva_reports if 'rmTimeiou' in item ]
    eva_report_NO_rmTimeiou = [item for item in eva_reports if 'rmTimeiou' not in item ]

    eva_report_NO_rmTimeiou.extend(eva_report_rmTimeiou)
    eva_reports = eva_report_NO_rmTimeiou

    acc_table, acc_table_IOU_version = read_accuracy_multi_reports(eva_reports)


    acc_table_pd = pd.DataFrame(acc_table)
    acc_table_IOU_version_pd = pd.DataFrame(acc_table_IOU_version)

    with pd.ExcelWriter(output_file) as writer:
        acc_table_pd.to_excel(writer, sheet_name='accuracy table')
        acc_table_IOU_version_pd.to_excel(writer, sheet_name='accuracy table IOU version')
        # set format
        workbook = writer.book
        format = workbook.add_format({'num_format': '#0.000'})
        acc_talbe_sheet = writer.sheets['accuracy table']
        acc_talbe_sheet.set_column('G:I',None,format)
        acc_iou_talbe_sheet = writer.sheets['accuracy table IOU version']
        acc_iou_talbe_sheet.set_column('G:I', None, format)


def main(options, args):

    eva_reports = [item for item in args]
    output_file = options.output
    eva_reports_to_table(eva_reports, output_file)

    pass


if __name__ == "__main__":
    usage = "usage: %prog [options] eva_report eva_report ... "
    parser = OptionParser(usage=usage, version="1.0 2020-6-28")
    parser.description = 'Introduction: convert the evaluation reports to tables'

    parser.add_option("-o", "--output",
                      action="store", dest="output",default="accuracy_table.xlsx",
                      help="the output file path")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)

