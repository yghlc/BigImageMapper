#!/usr/bin/env python
# Filename: analyze_dataAug_results 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 29 March, 2021
"""

import os, sys
code_dir = os.path.expanduser('~/codes/PycharmProjects/Landuse_DL')
sys.path.insert(0, code_dir)
import basic_src.io_function as io_function

import pandas as pd

def output_max_min_miou(pd_table):
    data_aug_options = pd_table['data_augmentation'].tolist()
    # train_class_0	train_class_1	val_class_0	val_class_1	class_1	overall time_total_h
    train_c0_count = pd_table['train_class_0'].tolist()
    train_c1_count = pd_table['train_class_1'].tolist()
    val_c0_count = pd_table['val_class_0'].tolist()
    val_c1_count = pd_table['val_class_1'].tolist()
    mIOU_c1 = pd_table['class_1'].tolist()
    mIOU_overall = pd_table['overall'].tolist()
    total_time  = pd_table['time_total_h'].tolist()

    # find max and min mIOU for class_1
    max_index = mIOU_c1.index(max(mIOU_c1))
    print('class_1::: mIOU: %f, train_c0_num:%d, train_c1_num:%d, val_c0_count:%d, val_c1_count:%d,'
          'total_time_h: %f,augmentation option: %s, '%( mIOU_c1[max_index], train_c0_count[max_index], train_c1_count[max_index],
                                val_c0_count[max_index], val_c1_count[max_index], total_time[max_index],data_aug_options[max_index],))

    min_index = mIOU_c1.index(min(mIOU_c1))
    print('class_1::: mIOU: %f, train_c0_num:%d, train_c1_num:%d, val_c0_count:%d, val_c1_count:%d,'
          'total_time_h: %f, augmentation option: %s '%( mIOU_c1[min_index], train_c0_count[min_index], train_c1_count[min_index],
                                val_c0_count[min_index], val_c1_count[min_index], total_time[min_index],data_aug_options[min_index]))

    # find max and min mIOU for overall
    max_index = mIOU_overall.index(max(mIOU_overall))
    print('overall::: mIOU: %f, train_c0_num:%d, train_c1_num:%d, val_c0_count:%d, val_c1_count:%d,'
          'total_time_h: %f,augmentation option: %s, '%( mIOU_overall[max_index], train_c0_count[max_index], train_c1_count[max_index],
                                val_c0_count[max_index], val_c1_count[max_index], total_time[max_index],data_aug_options[max_index],))

    min_index = mIOU_overall.index(min(mIOU_overall))
    print('overall::: mIOU: %f, train_c0_num:%d, train_c1_num:%d, val_c0_count:%d, val_c1_count:%d,'
          'total_time_h: %f, augmentation option: %s '%( mIOU_overall[min_index], train_c0_count[min_index], train_c1_count[min_index],
                                val_c0_count[min_index], val_c1_count[min_index], total_time[min_index],data_aug_options[min_index]))

def output_miou_for_each_dataAug_options(pd_table):
    data_aug_options = pd_table['data_augmentation'].tolist()
    mIOU_c1 = pd_table['class_1'].tolist()
    mIOU_overall = pd_table['overall'].tolist()

    aug_options_c1 = {}
    aug_options_overall = {}

    for opt, miou_c1, miou_o in zip(data_aug_options, mIOU_c1, mIOU_overall):
        # print(opt, miou_c1, miou_o)
        opt_list = [item.strip() for item in opt.split(',')]
        for aug in opt_list:
            if aug in aug_options_c1.keys():
                aug_options_c1[aug].append(miou_c1)
            else:
                aug_options_c1[aug] = [miou_c1]

            if aug in aug_options_overall.keys():
                aug_options_overall[aug].append(miou_o)
            else:
                aug_options_overall[aug] = [miou_o]

    for key in aug_options_c1:
        value_ist = aug_options_c1[key]
        print('class_1: exp count: %d, mean, max, and min miou_c1: %f %f %f, aug option: %s'%
              (len(value_ist), sum(value_ist)/len(value_ist), max(value_ist), min(value_ist),key))

    for key in aug_options_overall:
        value_ist = aug_options_overall[key]
        print('overall: exp count: %d, mean, max, and min miou_c1: %f %f %f, aug option: %s'%
              (len(value_ist), sum(value_ist)/len(value_ist), max(value_ist), min(value_ist),key))



def main():
    #
    dataAug_table = pd.read_excel(dataAug_res_WR)
    output_max_min_miou(dataAug_table)
    output_miou_for_each_dataAug_options(dataAug_table)

    pass

if __name__ == '__main__':

    # 255 experiments of data augmentation result, mIOU for class_1 and overall
    dataAug_res_WR = os.path.expanduser('~/Data/Arctic/canada_arctic/autoMapping/tune_dataAug_para_tesia.xlsx')

    # Apply each trained model (255 ones) to other images acquired in 2020 but different dates (not entire images,
    # but only the subImages extracted from training polygons)
    test_dataAug_res_WR_dir = os.path.expanduser('~/Data/Arctic/canada_arctic/autoMapping/eval_on_multi_datasets')
    test_dataAug_res_WR_list = io_function.get_file_list_by_ext('.xlsx',test_dataAug_res_WR_dir, bsub_folder=False)

    main()

    pass



