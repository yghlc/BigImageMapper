#!/usr/bin/env python
# Filename: hyper_para_ray 
"""
introduction: conduct data augmentation options testing using ray

run in :

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 16 March, 2021
"""

import os,sys
code_dir = os.path.expanduser('~/codes/PycharmProjects/Landuse_DL')
sys.path.insert(0, code_dir)
import basic_src.io_function as io_function

from ray import tune
from datetime import datetime
import pandas as pd

# backbones = ['deeplabv3plus_xception65.ini','deeplabv3plus_xception41.ini','deeplabv3plus_xception71.ini',
#             'deeplabv3plus_resnet_v1_50_beta.ini','deeplabv3plus_resnet_v1_101_beta.ini',
#             'deeplabv3plus_mobilenetv2_coco_voc_trainval.ini','deeplabv3plus_mobilenetv3_large_cityscapes_trainfine.ini',
#             'deeplabv3plus_mobilenetv3_small_cityscapes_trainfine.ini','deeplabv3plus_EdgeTPU-DeepLab.ini']



# template para (contain para_files) splited image patches before augmented
data_ini_dir=os.path.expanduser('~/Data/Arctic/canada_arctic/autoMapping/WR_dataAug_test')

from hyper_para_ray import modify_parameter
# from hyper_para_ray import get_total_F1score

def get_augment_options():
    from itertools import combinations

    # test_id = 0
    img_aug_options = []
    for count in range(1, 9):
        comb = combinations(['flip', 'blur', 'crop', 'scale', 'rotate', 'bright', 'contrast', 'noise'], count)
        for idx, img_aug in enumerate(list(comb)):
            # spaces are not allow in img_aug_str
            img_aug_str = ','.join(img_aug)
            img_aug_options.append(img_aug_str)

    io_function.save_list_to_txt('img_aug_str.txt',img_aug_options)

    return img_aug_options


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

def copy_data_ini_exe_files(org_dir, work_dir):

    res = os.system('cp -r %s/*  %s/.'%(org_dir,work_dir))
    if res != 0:
        sys.exit(1)

def objective_overall_miou(data_augmentation):

    sys.path.insert(0, code_dir)
    sys.path.insert(0, os.path.join(code_dir,'workflow'))   # for some module in workflow folder
    import basic_src.io_function as io_function
    import parameters
    import workflow.whole_procedure as whole_procedure


    para_file = 'main_para_dataAug.ini'
    work_dir = os.getcwd()

    # create a training folder
    copy_data_ini_exe_files(data_ini_dir,work_dir)

    exp_name = parameters.get_string_parameters(para_file,'expr_name')

    # change para_file
    modify_parameter(os.path.join(work_dir, para_file),'data_augmentation',data_augmentation)


    # run training
    # whole_procedure.run_whole_procedure(para_file, b_train_only=True)
    res = os.system('exe_tesia.sh')
    if res != 0:
        sys.exit(1)

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
    data_augmentation = config['data_augmentation']
    overall_miou = objective_overall_miou(data_augmentation)
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

    loc_dir = "./ray_results"
    tune_name = "tune_dataAug_para_tesia"
    augment_options = get_augment_options()
    file_folders = io_function.get_file_list_by_pattern(os.path.join(loc_dir, tune_name),'*')
    if len(file_folders) > 1:
        b_resume = True
    else:
        b_resume = False

    analysis = tune.run(
        training_function,
        resources_per_trial={"gpu": 1}, # use one GPUs,
        local_dir=loc_dir,
        name=tune_name,
        # fail_fast=True,     # Stopping after the first failure
        log_to_file=("stdout.log", "stderr.log"),     #Redirecting stdout and stderr to files
        trial_name_creator=tune.function(trial_name_string),
        trial_dirname_creator=tune.function(trial_dir_string),
        resume=b_resume,
        config={
            "data_augmentation": tune.grid_search(augment_options)
        }

        )

    print("Best config: ", analysis.get_best_config(
        metric="overall_miou", mode="max"))

    # Get a dataframe for analyzing trial results.
    df = analysis.results_df
    output_file = 'training_dataAug_ray_tune_%s.xlsx'%(datetime.now().strftime("%Y%m%d_%H%M%S"))
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

