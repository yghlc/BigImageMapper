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

def find_info_realted_to_train_dir(train_val_table, train_dir, info_key):
    folder_list = train_val_table['folder'].tolist()
    info_list = train_val_table[info_key].tolist()
    for dir, info in zip(folder_list, info_list):
        if dir == train_dir:
            return info

    return None

def output_mean_max_miou_all_test_data(test_xlsx_list, train_val_table):

    mean_miou_c1_each_test = {}
    max_miou_c1_each_test = {}
    max_miou_c1_test_aug_options = {}
    for xlsx in test_xlsx_list:
        print(xlsx)
        test_pd_table = pd.read_excel(xlsx)
        miou_c1_list = test_pd_table['class_1'].tolist()
        train_dir_list = test_pd_table['train_dir'].tolist()
        key = os.path.splitext(os.path.basename(xlsx))[0]
        mean_miou_c1_each_test[key] = sum(miou_c1_list)/len(miou_c1_list)
        max_miou_c1_each_test[key] = max(miou_c1_list)

        # get trianing_dir
        max_idx = miou_c1_list.index(max_miou_c1_each_test[key])
        train_dir = train_dir_list[max_idx]
        data_aug_options = find_info_realted_to_train_dir(train_val_table,train_dir,'data_augmentation')
        max_miou_c1_test_aug_options[key] = data_aug_options

    key_list = list(mean_miou_c1_each_test.keys())
    key_list.sort()
    mean_list = []
    max_list = []
    aug_option_list = []
    for key in key_list:
        print('%s mean miou c1: %f, max miou c1: %f'%(key, mean_miou_c1_each_test[key], max_miou_c1_each_test[key]))
        mean_list.append(mean_miou_c1_each_test[key])
        max_list.append(max_miou_c1_each_test[key])
        aug_option_list.append(max_miou_c1_test_aug_options[key])


    # data augmentation count:
    data_option_count = {}
    for key in key_list:
        opt_list = [ item.strip() for item in max_miou_c1_test_aug_options[key].split(',')]
        for opt in opt_list:
            if opt in data_option_count.keys():
                data_option_count[opt] += 1
            else:
                data_option_count[opt] = 1

    print(data_option_count)

    save_dict = {'test_images':key_list, 'mean_miou_class_1':mean_list,
               'max_miou_class_1':max_list, 'max_miou_aug_options':aug_option_list}

    save_dict_pd = pd.DataFrame(save_dict)
    with pd.ExcelWriter('miou_mean_max_test_data.xlsx') as writer:
        save_dict_pd.to_excel(writer, sheet_name='table')

    print("save to %s"%'miou_mean_max_test_data.xlsx')

def main():
    # miou for the validation data (10%)
    dataAug_table = pd.read_excel(dataAug_res_WR)
    # output_max_min_miou(dataAug_table)
    # output_miou_for_each_dataAug_options(dataAug_table)

    # miou for test data (different dates)
    output_mean_max_miou_all_test_data(test_dataAug_res_WR_list,dataAug_table)


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



