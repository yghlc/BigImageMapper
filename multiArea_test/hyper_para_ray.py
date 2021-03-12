#!/usr/bin/env python
# Filename: hyper_para_ray 
"""
introduction: conduct hyper-parameter testing using ray

run in : ~/Data/Arctic/canada_arctic/autoMapping

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 22 February, 2021
"""

import os,sys
code_dir = os.path.expanduser('~/codes/PycharmProjects/Landuse_DL')


from ray import tune

# backbones = ['deeplabv3plus_xception65.ini','deeplabv3plus_xception41.ini','deeplabv3plus_xception71.ini',
#             'deeplabv3plus_resnet_v1_50_beta.ini','deeplabv3plus_resnet_v1_101_beta.ini',
#             'deeplabv3plus_mobilenetv2_coco_voc_trainval.ini','deeplabv3plus_mobilenetv3_large_cityscapes_trainfine.ini',
#             'deeplabv3plus_mobilenetv3_small_cityscapes_trainfine.ini','deeplabv3plus_EdgeTPU-DeepLab.ini']

backbones = ['deeplabv3plus_xception65.ini']

area_ini_list = ['area_Willow_River.ini','area_Banks_east_nirGB.ini','area_Ellesmere_Island_nirGB.ini']

# template para (contain para_files)
ini_dir='ini_files'

def copy_ini_files(ini_dir, work_dir, para_file, area_ini_list,backbone):

    import basic_src.io_function as io_function
    ini_list = [para_file, backbone]
    ini_list.extend(area_ini_list)
    for ini in ini_list:
        io_function.copy_file_to_dst(os.path.join(ini_dir, ini ), os.path.join(work_dir,ini))

def modify_parameter(para_file, para_name, new_value):

    import parameters
    parameters.write_Parameters_file(para_file,para_name,new_value)

def get_total_F1score(working_dir):

    import basic_src.io_function as io_function
    from utility.eva_report_to_tables import read_accuracy_multi_reports

    reports = io_function.get_file_list_by_pattern(working_dir,'result_backup/*/*eva_report*.txt')
    acc_table, acc_table_IOU_version = read_accuracy_multi_reports(reports)
    total_tp = sum(acc_table['TP'])
    total_fp = sum(acc_table['FP'])
    total_fn = sum(acc_table['FN'])

    precision = float(total_tp) / (float(total_tp) + float(total_fp))
    recall = float(total_tp) / (float(total_tp) + float(total_fn))
    if (total_tp > 0):
        F1score = 2.0 * precision * recall / (precision + recall)
    else:
        F1score = 0
    return F1score

def objective_total_F1(lr, iter_num,batch_size,backbone):

    sys.path.insert(0, code_dir)
    import basic_src.io_function as io_function
    import workflow.whole_procedure as whole_procedure

    para_file = 'main_para_3Area.ini'
    # ray tune will change current folder to its logdir, change it back
    os.chdir(curr_dir_before_ray)
    print('\n\n\n current folder',os.getcwd(),'\n\n\n')


    # create a training folder
    work_dir = 'hyper'+'-lr%.5f'%lr + '-iter%d'%iter_num + '-batch%d'%batch_size + '-%s'%os.path.splitext(backbone)[0]
    if os.path.isdir(work_dir) is False:
        io_function.mkdir(work_dir)
    copy_ini_files(ini_dir,work_dir,para_file,area_ini_list,backbone)

    # change para_file
    modify_parameter(os.path.join(work_dir, para_file),'network_setting_ini',backbone)
    modify_parameter(os.path.join(work_dir, backbone),'base_learning_rate',lr)
    modify_parameter(os.path.join(work_dir, backbone),'batch_size',batch_size)
    modify_parameter(os.path.join(work_dir, backbone),'iteration_num',iter_num)

    # run training
    whole_procedure.run_whole_procedure(para_file,working_dir=work_dir)

    # remove files to save storage
    os.system('rm -rf %s/exp*'%work_dir)
    os.system('rm -rf %s/multi_inf_results'%work_dir)
    os.system('rm -rf %s/split*'%work_dir)
    os.system('rm -rf %s/sub*'%work_dir)
    os.system('rm -rf %s/tfrecord*'%work_dir)

    # calculate the F1 score across all regions (total F1)
    totalF1 = get_total_F1score(work_dir)
    return totalF1


def training_function(config,checkpoint_dir=None):
    # Hyperparameters
    lr, iter_num,batch_size,backbone = config["lr"], config["iter_num"],config["batch_size"],config["backbone"]

    total_F1_score = objective_total_F1(lr, iter_num,batch_size,backbone)

    # Feed the score back back to Tune.
    tune.report(total_F1=total_F1_score)

# tune.choice([1])  # randomly chose one value
if __name__ == '__main__':
    
    curr_dir_before_ray = os.getcwd()
    print('\n\ncurrent folder before ray tune: ', curr_dir_before_ray, '\n\n')

    # for the user defined moduel in code_dir, need to be imported in functions
    # sys.path.insert(0, code_dir)
    # import parameters
    # import basic_src.io_function as io_function
    # import workflow.whole_procedure as whole_procedure
    # from utility.eva_report_to_tables import read_accuracy_multi_reports


    analysis = tune.run(
        training_function,
        resources_per_trial={"gpu": 2}, # use two GPUs, 12 CPUs on tesia #"cpu": 14, don't limit cpu, eval.py will not use all
        local_dir="./ray_results",
        name="test_on_tesia",
        # fail_fast=True,     # Stopping after the first failure
        log_to_file=("stdout.log", "stderr.log"),     #Redirecting stdout and stderr to files
        config={
            "lr": tune.grid_search([0.0001,0.007, 0.014]),   # ,0.007, 0.014, 0.028,0.056
            "iter_num": tune.grid_search([30000]), # , 60000,90000
            "batch_size": tune.grid_search([8]), # 16, 32, 64, 128
            "backbone": tune.grid_search(backbones)
        })

    print("Best config: ", analysis.get_best_config(
        metric="total_F1", mode="max"))

    # Get a dataframe for analyzing trial results.
    df = analysis.results_df
