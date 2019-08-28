#!/usr/bin/env python
# Filename: acc_tables_to_latex 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 21 February, 2019
"""


import csv
import os,sys

csv_table = "img_aug_test_results_accuracy_table_to_latex.csv"
# csv_table = "img_aug_test_results_accuracy_table_noPost_to_latex.csv"

def format_float(str_list):

    if len(str_list) != 7:
        raise ValueError('the count is not 7')

    out_str = []
    out_str.append('%.1f' %float(str_list[0]))  # iou threshold
    out_str.extend([ item for item in str_list[1:4] ]) # TP, FP, FN
    out_str.extend(['%.3f' % float(item) for item in str_list[4:7]] )  # precision, recall, F1 score

    return out_str

def output_one_test(idx,lines):

    print('\midrule')
    line_1 = lines[idx]
    method = line_1[1]  # flip, blur, crop, scale, rotate
    m_labels = []
    if 'flip' in method: m_labels.append('F')
    if 'blur' in method: m_labels.append('B')
    if 'crop' in method: m_labels.append('C')
    if 'scale' in method: m_labels.append('S')
    if 'rotate' in method: m_labels.append('R')

    # test_num = idx // 3 + 2
    test_num = line_1[-1] // 3 + 1

    # to latex
    tmp='\multirow{3}{*}{%d} &  \multirow{3}{*}{%s} & \multirow{3}{*}{%s} & \multirow{3}{*}{%s} & \multirow{3}{*}{%s} &' \
        % (test_num, ', '.join(m_labels), '%.3f'%float(line_1[2]), line_1[3], line_1[4])

    # to word
    # tmp='%d & %s & %s & %s & %s &' \
    #     % (test_num, ', '.join(m_labels), line_1[2], line_1[3], line_1[4])

    out_str_1 = format_float(line_1[5:-1])
    print(tmp + ' & '.join(out_str_1) + '\\\\[-1ex]' )

# '&0.8&112&97&90&0.536 &0.555 &0.545 \\\\'

    line_2 = lines[idx + 1]
    out_str_2 = format_float(line_2[5:-1])
    print(' &  & &  &   & ' +
          ' & '.join(out_str_2) + '\\\\[-1ex]')

    line_3 = lines[idx + 2]
    out_str_3 = format_float(line_3[5:-1])
    print(' &  & &  &   & ' +
          ' & '.join(out_str_3) + '\\\\[-1ex]')


    pass

def data_augmentation_statistics(lines):
    '''
    output the statistic information of different data augmentation method
    :return: 
    '''

    count_dict = {'flip':0, 'blur':0, 'crop':0, 'scale':0, 'rotate':0}
    iou_8_dict = {'flip':[], 'blur':[], 'crop':[], 'scale':[], 'rotate':[]}
    iou_4_dict = {'flip':[], 'blur':[], 'crop':[], 'scale':[], 'rotate':[]}
    iou_0_dict = {'flip':[], 'blur':[], 'crop':[], 'scale':[], 'rotate':[]}

    ## output all
    line_count = len(lines)
    idx = 1

    key_short = {'flip':'F','blur':'B','crop':'C','scale':'S','rotate':'R'}

    while idx < line_count:
        # output_one_test(idx, lines)

        line_1 = lines[idx]
        line_2 = lines[idx + 1]
        line_3 = lines[idx + 2]
        method = line_1[1]  # flip, blur, crop, scale, rotate
        m_labels = []
        # print(method)
        if 'flip' in method:
            count_dict['flip'] += 1
            iou_8_dict['flip'].append(float(line_1[11]))    # f1 score
            iou_4_dict['flip'].append(float(line_2[11]))
            iou_0_dict['flip'].append(float(line_3[11]))

        if 'blur' in method:
            count_dict['blur'] += 1
            iou_8_dict['blur'].append(float(line_1[11]))    # f1 score
            iou_4_dict['blur'].append(float(line_2[11]))
            iou_0_dict['blur'].append(float(line_3[11]))

        if 'crop' in method:
            count_dict['crop'] += 1
            iou_8_dict['crop'].append(float(line_1[11]))    # f1 score
            iou_4_dict['crop'].append(float(line_2[11]))
            iou_0_dict['crop'].append(float(line_3[11]))

        if 'scale' in method:
            count_dict['scale'] += 1
            iou_8_dict['scale'].append(float(line_1[11]))    # f1 score
            iou_4_dict['scale'].append(float(line_2[11]))
            iou_0_dict['scale'].append(float(line_3[11]))

        if 'rotate' in method:
            count_dict['rotate'] += 1
            iou_8_dict['rotate'].append(float(line_1[11]))    # f1 score
            iou_4_dict['rotate'].append(float(line_2[11]))
            iou_0_dict['rotate'].append(float(line_3[11]))

        idx += 3

    print(count_dict)
    print(iou_8_dict)
    print(iou_4_dict)
    print(iou_0_dict)

    # get min, max, min f1 score
    print('min , max  , mean')
    print('iou: 0.8')
    for key in iou_8_dict.keys():
        max_value = max(iou_8_dict[key])
        min_value = min(iou_8_dict[key])
        avg_value = sum(iou_8_dict[key]) / len(iou_8_dict[key])
        # print('%6s, min: %.3f, max: %.3f, mean: %.3f'%(key,min_value,max_value,avg_value))
        print('& %6s & %.3f & %.3f & %.3f \\\\' % (key_short[key], min_value, max_value, avg_value))

    print('iou: 0.4')
    for key in iou_4_dict.keys():
        max_value = max(iou_4_dict[key])
        min_value = min(iou_4_dict[key])
        avg_value = sum(iou_4_dict[key]) / len(iou_4_dict[key])
        print('& %6s & %.3f & %.3f & %.3f \\\\'%(key_short[key],min_value,max_value,avg_value))

    print('iou: 0.0')
    for key in iou_0_dict.keys():
        max_value = max(iou_0_dict[key])
        min_value = min(iou_0_dict[key])
        avg_value = sum(iou_0_dict[key]) / len(iou_0_dict[key])
        print('& %6s & %.3f & %.3f & %.3f \\\\'%(key_short[key],min_value,max_value,avg_value))





    pass

with open(csv_table) as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')

    lines = [row for row in spamreader]
    for idx,row in enumerate(lines): # insert row number, for output test number after sort
        row.append(idx)
    line_count = len(lines)
    print(line_count)

    # data_augmentation_statistics(lines)

    ## output all
    # idx = 1
    # while idx < line_count:
    #     output_one_test(idx, lines)
    #     idx += 3
    #
    # remove the first line
    del lines[0]
    # for line in lines:
    #     print(float(line[2]))

    # sort
    lines.sort(key=lambda x: float(x[2]),reverse=True)   # x[2] is the ap value, descending

    for line in lines:
        print(float(line[2]))

    #get top 5 of AP
    idx = 0
    for k in range(0,5):
        # print(idx)
        output_one_test(idx, lines)
        idx += 3

    # get bottom 5
    idx = len(lines) - 3*5
    for k in range(0,5):
        # print(idx)
        output_one_test(idx, lines)
        idx += 3




