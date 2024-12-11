#!/usr/bin/env python
# Filename: hyper_para_ray 
"""
introduction: conduct hyper-parameter tuning using ray: for tuning the parameters of CLIP models

run in :

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 11 December, 2024
"""

import os,sys
code_dir = os.path.expanduser('~/codes/PycharmProjects/BigImageMapper')

sys.path.insert(0, code_dir)
import basic_src.io_function as io_function
import basic_src.basic as basic
import parameters


backbones = ['deeplabv3plus_xception65.ini','deeplabv3plus_xception41.ini','deeplabv3plus_xception71.ini',
            'deeplabv3plus_resnet_v1_50_beta.ini','deeplabv3plus_resnet_v1_101_beta.ini',
            'deeplabv3plus_mobilenetv2_coco_voc_trainval.ini','deeplabv3plus_mobilenetv3_large_cityscapes_trainfine.ini',
            'deeplabv3plus_mobilenetv3_small_cityscapes_trainfine.ini','deeplabv3plus_EdgeTPU-DeepLab.ini']

# backbones = ['deeplabv3plus_xception65.ini']

area_ini_list = ['area_Willow_River.ini', 'area_Banks_east.ini', 'area_Ellesmere_Island.ini',
                 'area_Willow_River_nirGB.ini','area_Banks_east_nirGB.ini','area_Ellesmere_Island_nirGB.ini']


# template para (contain para_files)
ini_dir=os.path.expanduser('~/Data/Arctic/canada_arctic/autoMapping/ini_files')
training_data_dir = os.path.expanduser('~/Data/Arctic/canada_arctic/autoMapping/training_find_tune_data')

# from hyper_para_ray import modify_parameter
# from hyper_para_ray import get_total_F1score


def get_CPU_GPU_counts():
    import GPUtil
    # Get all available GPUs
    gpus = GPUtil.getGPUs()
    # Get the number of GPUs
    gpu_count = len(gpus)

    # Get the number of logical CPUs
    cpu_count = os.cpu_count()

    return cpu_count, gpu_count


def modify_parameter(para_file, para_name, new_value):
    parameters.write_Parameters_file(para_file,para_name,new_value)


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
    basic.outputlogMessage('trial_id: %s'%trial_id)
    # basic.outputlogMessage('experiment_tag: %s'%experiment_tag)
    return 'clip_tuning_' + str(trial_id)[-6:]  # should not write as [-6:-1], use the last 5 digits + '_'.

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

def copy_training_datas(data_dir, work_dir):
    # for the case, we have same parameter for preparing data, then just copy the data to save time.
    # buffer_size, training_data_per, data_augmentation, data_aug_ignore_classes

    sub_files = ['sub', 'split', 'list','tfrecord']
    for sub_str in sub_files:
        command_str = 'cp -r %s/%s*  %s/.'%(data_dir,sub_str, work_dir)
        print(command_str)
        res = os.system(command_str)
        if res != 0:
            sys.exit(1)
    return

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

    exp_name = parameters.get_string_parameters(para_file, 'expr_name')

    # check if ray create a new folder because previous run dir already exist.
    # e.g., previous dir: "multiArea_deeplabv3P_00000", new dir: "multiArea_deeplabv3P_00000_ce9a"
    # then read the miou directly
    pre_work_dir = work_dir[:-5]  # remove something like _ce9a
    if os.path.isdir(pre_work_dir):
        iou_path = os.path.join(pre_work_dir, exp_name, 'eval', 'miou.txt')
        overall_miou = get_overall_miou(iou_path)
        return overall_miou

    # don't initialize the last layer when using these backbones
    if 'mobilenetv2' in backbone or 'mobilenetv3' in backbone or 'EdgeTPU' in backbone:
        modify_parameter(os.path.join(work_dir, para_file), 'b_initialize_last_layer', 'No')

    # change para_file
    modify_parameter(os.path.join(work_dir, para_file),'network_setting_ini',backbone)
    modify_parameter(os.path.join(work_dir, backbone),'base_learning_rate',lr)
    modify_parameter(os.path.join(work_dir, backbone),'batch_size',batch_size)
    modify_parameter(os.path.join(work_dir, backbone),'iteration_num',iter_num)

    modify_parameter(os.path.join(work_dir, para_file),'buffer_size',buffer_size)
    modify_parameter(os.path.join(work_dir, para_file),'training_data_per',training_data_per)
    modify_parameter(os.path.join(work_dir, para_file),'data_augmentation',data_augmentation)
    modify_parameter(os.path.join(work_dir, para_file),'data_aug_ignore_classes',data_aug_ignore_classes)

    # for the cases, we have same parameter for preparing data, then just copy the data to save time.
    copy_training_datas(training_data_dir,work_dir)

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
    # lr, iter_num,batch_size,backbone,buffer_size,training_data_per,data_augmentation,data_aug_ignore_classes = \
    #     config["lr"], config["iter_num"],config["batch_size"],config["backbone"],config['buffer_size'],\
    #     config['training_data_per'],config['data_augmentation'],config['data_aug_ignore_classes']

    # Hyperparameters
    lr = config["lr"]
    iter_num = config["iter_num"]
    batch_size = config["batch_size"]
    backbone = config["backbone"]
    buffer_size = config["buffer_size"]
    training_data_per = config["training_data_per"]
    data_augmentation = config["data_augmentation"]
    data_aug_ignore_classes = config["data_aug_ignore_classes"]

    # # Restore from checkpoint if provided
    # if checkpoint_dir:
    #     checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pkl")
    #     if os.path.exists(checkpoint_path):
    #         with open(checkpoint_path, "rb") as f:
    #             checkpoint_data = pickle.load(f)
    #             print(f"Restored checkpoint: {checkpoint_data}")
    #             # Restore any state here (e.g., model weights or iteration count)

    # # Perform training (or simulate training here)
    # overall_miou = objective_overall_miou(
    #     lr, iter_num, batch_size, backbone, buffer_size,
    #     training_data_per, data_augmentation, data_aug_ignore_classes
    # )

    # For debugging, generate a random number to simulate the "overall_miou" metric
    overall_miou = random.uniform(0, 1)  # Random value between 0 and 1
    print(f"Generated random overall_miou: {overall_miou}")


    # # Save a checkpoint
    # with tune.checkpoint_dir(step=iter_num) as checkpoint_dir:
    #     checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pkl")
    #     with open(checkpoint_path, "wb") as f:
    #         pickle.dump({"lr": lr, "iter_num": iter_num}, f)  # Save any necessary state

    # overall_miou = objective_overall_miou(lr, iter_num,batch_size,backbone,buffer_size,training_data_per,data_augmentation,data_aug_ignore_classes)

    # Feed the score back back to Tune.
    session.report({"overall_miou": overall_miou})

def stop_function(trial_id, result):
    # it turns out that stop_function it to check weather to run more experiments, not to decide whether run one experiment.
    pass

    # exp_folder =  'multiArea_deeplabv3P' + str(trial_id)[-6:]
    # # exp_dir = os.path.join(loc_dir, tune_name,exp_folder)
    # if os.path.isdir(exp_dir):
    #     print("%s exists, skip this experiment"%exp_dir)
    #     return True
    #
    # return False

# tune.choice([1])  # randomly chose one value

def main():

    # for the user defined module in code_dir, need to be imported in functions
    # sys.path.insert(0, code_dir)
    # import parameters
    # import basic_src.io_function as io_function
    # import workflow.whole_procedure as whole_procedure
    # from utility.eva_report_to_tables import read_accuracy_multi_reports

    loc_dir = "./ray_results"
    storage_path = os.path.abspath(loc_dir)
    tune_name = "tune_clip_para"

    # if os.path.isdir(loc_dir):
    #     io_function.mkdir(loc_dir)

    # Check if there are existing folders in the tuning directory
    file_folders = io_function.get_file_list_by_pattern(os.path.join(loc_dir, tune_name), '*')
    b_resume = len(file_folders) > 1

    # Define the search space (same as before)
    param_space = {
        "lr": tune.grid_search([0.007, 0.014, 0.021, 0.28]),
        "iter_num": tune.grid_search([30000]),  # , 60000, 90000
        "batch_size": tune.grid_search([8, 16, 32, 48, 96]),
        "backbone": tune.grid_search(backbones),
        "buffer_size": tune.grid_search([300]),  # 600
        "training_data_per": tune.grid_search([0.9]),
        "data_augmentation": tune.grid_search(['blur,crop,bright,contrast,noise']),
        'data_aug_ignore_classes': tune.grid_search(['class_0']),
    }

    cpu_count, gpu_count = get_CPU_GPU_counts()
    # Wrap the training function with resource requirements
    trainable = tune.with_resources(
        training_function,
        resources={"cpu": cpu_count, "gpu": gpu_count}  # Allocate 24 CPUs and 1 GPU per trial
    )


    # Configure the Tuner
    tuner = Tuner(
        trainable=trainable,
        param_space=param_space,
        tune_config=TuneConfig(
            metric="overall_miou",  # Metric to optimize
            mode="max",  # Maximize the metric
            num_samples=1,  # Number of samples (can be tuned as needed)
            scheduler=ASHAScheduler(),  # ASHA scheduler
            reuse_actors=b_resume,  # Resume trials if possible
        ),
        run_config=RunConfig(
            storage_path= f"file://{storage_path}",  # Directory to save results
            name=tune_name,  # Experiment name
            log_to_file=("stdout.log", "stderr.log"),  # Redirect logs
            # trial_name_creator=trial_name_string,  # Custom trial name
            # trial_dirname_creator=trial_dir_string,  # Custom trial directory name
        ),
    )

    # Run the tuner
    results = tuner.fit()

    # Get the best configuration
    best_result = results.get_best_result(metric="overall_miou", mode="max")
    print("Best config: ", best_result.config)

    # Get a DataFrame for analyzing trial results
    df = results.get_dataframe()
    output_file = f'training_miou_ray_tune_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
    with pd.ExcelWriter(output_file) as writer:
        df.to_excel(writer)
        print(f'Wrote trial results to {output_file}')



if __name__ == '__main__':

    import pandas as pd
    from datetime import datetime
    import ray
    from ray import tune
    from ray.tune import Tuner, TuneConfig
    from ray.air import RunConfig, session
    from ray.tune.schedulers import ASHAScheduler
    import pickle  # Add this import to use pickle for saving/loading checkpoints
    import random

    curr_dir_before_ray = os.getcwd()
    print('\n\ncurrent folder before ray tune: ', curr_dir_before_ray, '\n\n')
    main()

