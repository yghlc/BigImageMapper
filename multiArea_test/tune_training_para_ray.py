#!/usr/bin/env python
# Filename: hyper_para_ray 
"""
introduction: conduct hyper-parameter testing using ray

run in :

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 10 March, 2021
"""

import os,sys
code_dir = os.path.expanduser('~/codes/PycharmProjects/Landuse_DL')

from ray import tune
from datetime import datetime
import pandas as pd

# backbones = ['deeplabv3plus_xception65.ini','deeplabv3plus_xception41.ini','deeplabv3plus_xception71.ini',
#             'deeplabv3plus_resnet_v1_50_beta.ini','deeplabv3plus_resnet_v1_101_beta.ini',
#             'deeplabv3plus_mobilenetv2_coco_voc_trainval.ini','deeplabv3plus_mobilenetv3_large_cityscapes_trainfine.ini',
#             'deeplabv3plus_mobilenetv3_small_cityscapes_trainfine.ini','deeplabv3plus_EdgeTPU-DeepLab.ini']

backbones = ['deeplabv3plus_xception65.ini']

area_ini_list = ['area_Willow_River.ini', 'area_Banks_east.ini', 'area_Ellesmere_Island.ini',
                 'area_Willow_River_nirGB.ini','area_Banks_east_nirGB.ini','area_Ellesmere_Island_nirGB.ini']


# template para (contain para_files)
ini_dir=os.path.expanduser('~/Data/Arctic/canada_arctic/autoMapping/ini_files')

from hyper_para_ray import modify_parameter
# from hyper_para_ray import get_total_F1score

def trial_name_string(trial):
    """
    Args:
        trial (Trial): A generated trial object.

    Returns:
        trial_name (str): String representation of Trial.
    """
    print('\n\n trial_name_string:\n',trial,'\n\n')
    return str(trial)

def trial_dir_string(trial_id):
    # print('\n\n trial_dir_string:\n',trial,'\n\n')
    # return str(trial)   # should able to have more control on the dirname
    return 'multiArea_deeplabv3P' + str(trial_id)[-6:]  # should not write as [-6:-1], use the last 5 digits + '_'.

def get_overall_miou(miou_path):
    import basic_src.io_function as io_function
    # exp8/eval/miou.txt
    iou_dict = io_function.read_dict_from_txt_json(miou_path)
    return iou_dict['overall'][-1]

def copy_ini_files(ini_dir, work_dir, para_file, area_ini_list,backbone):

    import basic_src.io_function as io_function
    ini_list = [para_file, backbone]
    ini_list.extend(area_ini_list)
    for ini in ini_list:
        io_function.copy_file_to_dst(os.path.join(ini_dir, ini ), os.path.join(work_dir,ini), overwrite=True)

def objective_overall_miou(lr, iter_num,batch_size,backbone,buffer_size,training_data_per,data_augmentation,data_aug_ignore_classes):

    sys.path.insert(0, code_dir)
    sys.path.insert(0, os.path.join(code_dir,'workflow'))   # for some module in workflow folder
    import basic_src.io_function as io_function
    import parameters
    import workflow.whole_procedure as whole_procedure


    para_file = 'main_para_exp9.ini'
    work_dir = os.getcwd()

    # create a training folder
    copy_ini_files(ini_dir,work_dir,para_file,area_ini_list,backbone)

    exp_name = parameters.get_string_parameters(para_file,'expr_name')

    # change para_file
    modify_parameter(os.path.join(work_dir, para_file),'network_setting_ini',backbone)
    modify_parameter(os.path.join(work_dir, backbone),'base_learning_rate',lr)
    modify_parameter(os.path.join(work_dir, backbone),'batch_size',batch_size)
    modify_parameter(os.path.join(work_dir, backbone),'iteration_num',iter_num)

    modify_parameter(os.path.join(work_dir, para_file),'buffer_size',buffer_size)
    modify_parameter(os.path.join(work_dir, para_file),'training_data_per',training_data_per)
    modify_parameter(os.path.join(work_dir, para_file),'data_augmentation',data_augmentation)
    modify_parameter(os.path.join(work_dir, para_file),'data_aug_ignore_classes',data_aug_ignore_classes)

    # run training
    whole_procedure.run_whole_procedure(para_file, b_train_only=True)

    # remove files to save storage
    os.system('rm -rf %s/exp*/init_models'%work_dir)
    os.system('rm -rf %s/exp*/eval/events.out.tfevents*'%work_dir) # don't remove miou.txt
    # os.system('rm -rf %s/exp*/train'%work_dir)            # don't remove train folder
    os.system('rm -rf %s/exp*/vis'%work_dir)        # don't remove the export folder (for prediction)

    os.system('rm -rf %s/multi_inf_results'%work_dir)
    os.system('rm -rf %s/split*'%work_dir)
    os.system('rm -rf %s/sub*s'%work_dir)   # only subImages and subLabels
    os.system('rm -rf %s/sub*s_delete'%work_dir)   # only subImages_delete and subLabels_delete
    os.system('rm -rf %s/tfrecord*'%work_dir)

    iou_path = os.path.join(work_dir,exp_name,'eval','miou.txt')
    overall_miou = get_overall_miou(iou_path)
    return overall_miou


def training_function(config,checkpoint_dir=None):
    # Hyperparameters
    lr, iter_num,batch_size,backbone,buffer_size,training_data_per,data_augmentation,data_aug_ignore_classes = \
        config["lr"], config["iter_num"],config["batch_size"],config["backbone"],config['buffer_size'],\
        config['training_data_per'],config['data_augmentation'],config['data_aug_ignore_classes']

    overall_miou = objective_overall_miou(lr, iter_num,batch_size,backbone,buffer_size,training_data_per,data_augmentation,data_aug_ignore_classes)

    # Feed the score back back to Tune.
    tune.report(overall_miou=overall_miou)

# tune.choice([1])  # randomly chose one value

def main():

    # for the user defined module in code_dir, need to be imported in functions
    # sys.path.insert(0, code_dir)
    # import parameters
    # import basic_src.io_function as io_function
    # import workflow.whole_procedure as whole_procedure
    # from utility.eva_report_to_tables import read_accuracy_multi_reports


    analysis = tune.run(
        training_function,
        resources_per_trial={"gpu": 2}, # use two GPUs, 12 CPUs on tesia  # "cpu": 14, don't limit cpu, eval.py will not use all
        local_dir="./ray_results",
        name="tune_traning_para_tesia",
        # fail_fast=True,     # Stopping after the first failure
        log_to_file=("stdout.log", "stderr.log"),     #Redirecting stdout and stderr to files
        trial_name_creator=tune.function(trial_name_string),
        trial_dirname_creator=tune.function(trial_dir_string),
        resume=True,
        config={
            "lr": tune.grid_search([0.007, 0.014, 0.28]),   # ,0.007, 0.014, 0.028,0.056
            "iter_num": tune.grid_search([30000, 60000, 90000]), # , 60000,90000
            "batch_size": tune.grid_search([8,16,32]), # 16, 32, 64, 128
            "backbone": tune.grid_search(backbones),
            "buffer_size": tune.grid_search([300]),     # 600
            "training_data_per": tune.grid_search([0.9]),   #, 0.8
            "data_augmentation": tune.grid_search(['scale, bright, contrast, noise']),
            'data_aug_ignore_classes':tune.grid_search(['class_0',''])
        }
        # config={
        #     "lr": tune.grid_search([0.014]),   # ,0.007, 0.014, 0.028,0.056
        #     "iter_num": tune.grid_search([30000]), # , 60000,90000
        #     "batch_size": tune.grid_search([8]), # 16, 32, 64, 128
        #     "backbone": tune.grid_search(backbones),
        #     "buffer_size": tune.grid_search([300]),
        #     "training_data_per": tune.grid_search([0.9]),
        #     "data_augmentation": tune.grid_search(['scale, bright, contrast, noise']),
        #     'data_aug_ignore_classes':tune.grid_search(['class_0',''])
        # }
        
        
        )

    print("Best config: ", analysis.get_best_config(
        metric="overall_miou", mode="max"))

    # Get a dataframe for analyzing trial results.
    df = analysis.results_df
    output_file = 'training_miou_ray_tune_%s.xlsx'%(datetime.now().strftime("%Y%m%d_%H%M%S"))
    with pd.ExcelWriter(output_file) as writer:
        df.to_excel(writer)         # , sheet_name='accuracy table'
        # set format
        # workbook = writer.book
        # format = workbook.add_format({'num_format': '#0.000'})
        # acc_talbe_sheet = writer.sheets['accuracy table']
        # acc_talbe_sheet.set_column('G:I',None,format)
        print('write trial results to %s' % output_file)



if __name__ == '__main__':
    
    curr_dir_before_ray = os.getcwd()
    print('\n\ncurrent folder before ray tune: ', curr_dir_before_ray, '\n\n')
    main()

