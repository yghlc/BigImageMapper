#!/usr/bin/env python
# Filename: hyper_para_ray 
"""
introduction: conduct hyper-parameter testing using ray for post-processing

run in a mapping folder, such as: ~/Data/Arctic/canada_arctic/autoMapping/multiArea_deeplabV3+_5

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 4 March, 2021
"""

import os,sys
code_dir = os.path.expanduser('~/codes/PycharmProjects/Landuse_DL')

from ray import tune

from hyper_para_ray import modify_parameter
from hyper_para_ray import get_total_F1score


def postProcess_total_F1(minimum_area, min_slope, dem_diff_uplimit, dem_diff_buffer_size, IOU_threshold):

    sys.path.insert(0, code_dir)
    import basic_src.io_function as io_function
    import workflow.whole_procedure as whole_procedure

    para_file = 'main_para_postProc_tune.ini'
    # ray tune will change current folder to its logdir, change it back
    os.chdir(curr_dir_before_ray)
    print('\n\n\n current folder',os.getcwd(),'\n\n\n')
    work_dir = curr_dir_before_ray


    # create a training folder
    inf_post_note = str(minimum_area)+'_'+str(min_slope) +'_' + str(dem_diff_uplimit) +'_' + str(dem_diff_buffer_size) +'_' + str(IOU_threshold)

    # copy copy_ini_files
    io_function.copy_file_to_dst('main_para.ini', para_file,overwrite=True)

    # change para_file
    modify_parameter(para_file,'minimum_area',minimum_area)
    modify_parameter(para_file,'minimum_slope',min_slope)
    modify_parameter(para_file,'dem_difference_range', 'None,' +str(dem_diff_uplimit))
    modify_parameter(para_file,'buffer_size_dem_diff',dem_diff_buffer_size)
    modify_parameter(para_file,'IOU_threshold',IOU_threshold)

    # run training
    # whole_procedure.run_whole_procedure(para_file,working_dir=work_dir)
    whole_procedure.post_processing_backup(para_file,inf_post_note=inf_post_note)

    # calculate the F1 score across all regions (total F1)
    totalF1 = get_total_F1score(work_dir)
    return totalF1


def postProcess_function(config,checkpoint_dir=None):
    # Hyperparameters
    minimum_area, min_slope, dem_diff_uplimit, dem_diff_buffer_size, IOU_threshold = \
        config["minimum_area"], config["min_slope"], config["dem_diff_uplimit"], config["dem_diff_buffer_size"], config["IOU_threshold"]

    total_F1_score = postProcess_total_F1(minimum_area, min_slope, dem_diff_uplimit, dem_diff_buffer_size, IOU_threshold)

    # Feed the score back back to Tune.
    tune.report(total_F1=total_F1_score)


if __name__ == '__main__':

    curr_dir_before_ray = os.getcwd()
    print('\n\ncurrent folder before ray tune: ', curr_dir_before_ray, '\n\n')

    # for the user defined moduel in code_dir, need to be imported in functions
    # sys.path.insert(0, code_dir)
    # import parameters
    # import basic_src.io_function as io_function
    # import workflow.whole_procedure as whole_procedure
    # from utility.eva_report_to_tables import read_accuracy_multi_reports

    # tune.choice([1])  # randomly chose one value
    analysis = tune.run(
        postProcess_function,
        ## set CPU to 24, almost all CPU, to make sure each time only one process is run, because in the post-processing,
        ## many files are shared.
        resources_per_trial={"cpu": 24},
        local_dir="./ray_results",
        name="tune_parameters_for_postPorcessing",
        # fail_fast=True,     # Stopping after the first failure
        log_to_file=("stdout.log", "stderr.log"),     #Redirecting stdout and stderr to files
        config={
            "minimum_area": tune.grid_search([0, 90, 900, 2700]),    # 0 pixel, 10 pixel,100 pixel, 300 pixel
            "min_slope": tune.grid_search([0, 1,2]),
            "dem_diff_uplimit": tune.grid_search([-2, -1.5, -1, -0.5, 0]),
            "dem_diff_buffer_size": tune.grid_search([0, 20, 50, 100, 150, 200]),
            "IOU_threshold": tune.grid_search([0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        })



    print("Best config: ", analysis.get_best_config(
        metric="total_F1", mode="max"))

    # Get a dataframe for analyzing trial results.
    df = analysis.results_df
