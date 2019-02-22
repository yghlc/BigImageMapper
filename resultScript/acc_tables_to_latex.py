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
        % (test_num, ', '.join(m_labels), line_1[2], line_1[3], line_1[4])

    # to word
    # tmp='%d & %s & %s & %s & %s &' \
    #     % (test_num, ', '.join(m_labels), line_1[2], line_1[3], line_1[4])

    print(tmp + ' & '.join(line_1[5:-1]) + '\\\\' )

# '&0.8&112&97&90&0.536 &0.555 &0.545 \\\\'

    line_2 = lines[idx + 1]
    print(' &  & &  &   & ' +
          ' & '.join(line_2[5:-1]) + '\\\\')

    line_3 = lines[idx + 2]
    print(' &  & &  &   & ' +
          ' & '.join(line_3[5:-1]) + '\\\\')


    pass

with open(csv_table) as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')

    lines = [row for row in spamreader]
    for idx,row in enumerate(lines): # insert row number, for output test number after sort
        row.append(idx)
    line_count = len(lines)
    print(line_count)

    ## output all
    idx = 1
    while idx < line_count:
        output_one_test(idx, lines)
        idx += 3

    # remove the first line
    del lines[0]
    # for line in lines:
    #     print(float(line[2]))

    # sort
    lines.sort(key=lambda x: float(x[2]),reverse=True)   # x[2] is the ap value, descending

    for line in lines:
        print(float(line[2]))

    # get top 5 of AP
    # idx = 0
    # for k in range(0,5):
    #     # print(idx)
    #     output_one_test(idx, lines)
    #     idx += 3

    # get bottom 5
    # idx = len(lines) - 3*5
    # for k in range(0,5):
    #     # print(idx)
    #     output_one_test(idx, lines)
    #     idx += 3




