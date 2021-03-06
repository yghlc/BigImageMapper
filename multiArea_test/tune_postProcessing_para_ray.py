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
import pandas as pd
from datetime import datetime

from hyper_para_ray import modify_parameter
from hyper_para_ray import get_total_F1score

area_ini_list = ['area_Willow_River.ini','area_Banks_east_nirGB.ini','area_Ellesmere_Island_nirGB.ini']
backbones = ['deeplabv3plus_xception65.ini']
def trial_name_string(trial):
    """
    Args:
        trial (Trial): A generated trial object.

    Returns:
        trial_name (str): String representation of Trial.
    """
    print('\n\n trial_name_string:\n',trial,'\n\n')
    return str(trial)

def trial_dir_string(trial):
    print('\n\n trial_dir_string:\n',trial,'\n\n')
    return str(trial)   # should able to have more control on the dirname

def copy_original_mapped_polygons(curr_dir_before_ray,work_dir):
    # when ray start a process, we need to add code_dir again and import user-defined modules
    import basic_src.io_function as io_function
    org_dir = os.path.join(curr_dir_before_ray,'multi_inf_results')
    save_dir = os.path.join(work_dir,'multi_inf_results')

    shp_list = io_function.get_file_list_by_pattern(org_dir,'*/*.shp')
    shp_list = [ item for item in shp_list if 'post' not in os.path.basename(item)]   # remove 'post' ones
    for shp in shp_list:
        area_dir = os.path.join(save_dir, os.path.basename(os.path.dirname(shp)))
        if os.path.isdir(area_dir) is False:
            io_function.mkdir(area_dir)
        dst_path = os.path.join(area_dir, os.path.basename(shp))
        io_function.copy_shape_file(shp, dst_path)

def copy_ini_files(curr_dir_before_ray,work_dir):
    import basic_src.io_function as io_function
    area_ini_list.append('main_para.ini')
    area_ini_list.extend(backbones)
    for ini in area_ini_list:
        io_function.copy_file_to_dst(os.path.join(curr_dir_before_ray,ini), ini ,overwrite=True)


def postProcess_total_F1(minimum_area, min_slope, dem_diff_uplimit, dem_diff_buffer_size, min_demD_area,IOU_threshold):

    # when ray start a process, we need to add code_dir again and import user-defined modules
    sys.path.insert(0, code_dir)
    sys.path.insert(0, os.path.join(code_dir, 'workflow'))  # require in ray when import other modules in workflow folder
    import basic_src.io_function as io_function
    import workflow.whole_procedure as whole_procedure

    para_file = 'main_para.ini'
    # ray tune will change current folder to its logdir, change it back
    # os.chdir(curr_dir_before_ray)
    print('\n\n\n current folder',os.getcwd(),'\n\n\n')

    # allow ray to change current folder to its logdir, then we can run parallel
    work_dir = os.getcwd()
    copy_original_mapped_polygons(curr_dir_before_ray,work_dir)
    # work_dir = curr_dir_before_ray


    # create a training folder
    inf_post_note = str(minimum_area)+'_'+str(min_slope) +'_' + str(dem_diff_uplimit) +'_' + str(dem_diff_buffer_size) +'_' + str(IOU_threshold)

    # copy copy_ini_files
    copy_ini_files(curr_dir_before_ray,work_dir)
    

    # change para_file
    modify_parameter(para_file,'minimum_area',minimum_area)
    modify_parameter(para_file,'minimum_slope',min_slope)
    modify_parameter(para_file,'dem_difference_range', 'None,' +str(dem_diff_uplimit))
    modify_parameter(para_file,'buffer_size_dem_diff',dem_diff_buffer_size)
    modify_parameter(para_file,'minimum_dem_reduction_area',min_demD_area)
    modify_parameter(para_file,'IOU_threshold',IOU_threshold)

    # run training
    # whole_procedure.run_whole_procedure(para_file,working_dir=work_dir)
    test_id = 'multiArea_deeplabV3+_5_exp6'
    whole_procedure.post_processing_backup(para_file,inf_post_note=inf_post_note,b_skip_getshp=True,test_id=test_id)

    io_function.delete_file_or_dir('multi_inf_results') # remove a folder to save storage

    # calculate the F1 score across all regions (total F1)
    totalF1 = get_total_F1score(work_dir)
    return totalF1


def postProcess_function(config,checkpoint_dir=None):
    # Hyperparameters
    minimum_area, min_slope, dem_diff_uplimit, dem_diff_buffer_size, min_demD_area,  IOU_threshold = \
        config["minimum_area"], config["min_slope"], config["dem_diff_uplimit"], config["dem_diff_buffer_size"], config["minimum_dem_reduction_area"], config["IOU_threshold"]

    total_F1_score = postProcess_total_F1(minimum_area, min_slope, dem_diff_uplimit, dem_diff_buffer_size, min_demD_area,IOU_threshold)

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
        resources_per_trial={"cpu": 1},
        local_dir="./ray_results",
        name="postPorcessing",
        # fail_fast=True,     # Stopping after the first failure
        log_to_file=("stdout.log", "stderr.log"),     #Redirecting stdout and stderr to files
        trial_name_creator=tune.function(trial_name_string),
        trial_dirname_creator=tune.function(trial_dir_string),
        config={
            "minimum_area": tune.grid_search([0, 90, 900, 2700]),    # 0 pixel, 10 pixel,100 pixel, 300 pixel
            "min_slope": tune.grid_search([0, 1,2]),
            "dem_diff_uplimit": tune.grid_search([-2, -1.5, -1, -0.5, 0]),
            "dem_diff_buffer_size": tune.grid_search([0, 20, 50, 100, 150, 200]),
            "minimum_dem_reduction_area": tune.grid_search([0, 90, 900, 2700]),
            "IOU_threshold": tune.grid_search([0.5])   # 0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7
        }
        #     config={
        #     "minimum_area": tune.grid_search([900]),    # 0 pixel, 10 pixel,100 pixel, 300 pixel
        #     "min_slope": tune.grid_search([0]),
        #     "dem_diff_uplimit": tune.grid_search([-0.5]),
        #     "dem_diff_buffer_size": tune.grid_search([50]),
        #     "IOU_threshold": tune.grid_search([0.5])
        # }
        
        )



    print("Best config: ", analysis.get_best_config(
        metric="total_F1", mode="max"))

    # Get a dataframe for analyzing trial results.
    df = analysis.results_df
    output_file = 'post_processing_ray_tune_%s.xlsx'%(datetime.now().strftime("%Y%m%d_%H%M%S"))
    with pd.ExcelWriter(output_file) as writer:
        df.to_excel(writer)         # , sheet_name='accuracy table'
        # set format
        # workbook = writer.book
        # format = workbook.add_format({'num_format': '#0.000'})
        # acc_talbe_sheet = writer.sheets['accuracy table']
        # acc_talbe_sheet.set_column('G:I',None,format)
        print('write trial results to %s' % output_file)

