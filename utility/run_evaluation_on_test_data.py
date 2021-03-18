#!/usr/bin/env python
# Filename: run_evaluation_on_test_data 
"""
introduction: run evaluation for a test data using different trained model.

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 17 March, 2021
"""
import os, sys
from optparse import OptionParser

code_dir = os.path.expanduser('~/codes/PycharmProjects/Landuse_DL')
sys.path.insert(0, code_dir)

import deeplabBased.run_evaluation as run_evaluation

import basic_src.io_function as io_function
import basic_src.basic as basic
import parameters

import pandas as pd

from collect_training_results import get_miou_of_overall_and_class_1_step

def run_evaluation_multi_trained_models(train_root_dir,train_dir_pattern,para_file,output_file, working_dir=None):

    curr_dir = os.getcwd()
    if working_dir is None:
        working_dir = curr_dir

    os.chdir(working_dir)

    if os.path.isdir(train_root_dir) is False:
        raise ValueError('%s not exists'%train_root_dir)

    folder_list = io_function.get_file_list_by_pattern(train_root_dir,train_dir_pattern)
    folder_list = [item for item in folder_list if os.path.isdir(item) ]
    folder_list.sort()

    eval_output = {}
    eval_output['train_dir'] = []

    eval_output['class_1'] = []
    eval_output['overall'] = []
    eval_output['step'] = []


    for train_folder in folder_list:
        exp_name = parameters.get_string_parameters(para_file, 'expr_name')
        eval_dir = os.path.join(working_dir, exp_name, 'eval')
        bak_miou_dir = os.path.join(working_dir, exp_name, 'eval_%s' % os.path.basename(train_folder))
        if os.path.isdir(bak_miou_dir):
            basic.outputlogMessage('Evaluation on test data uisng model %s already exist, skip'%os.path.basename(train_folder))
            continue

        basic.outputlogMessage('run evaluation using trained model in %s'%train_folder)
        eval_output['train_dir'].append(os.path.basename(train_folder))

        # run evaluation
        TRAIN_LOGDIR = os.path.join(train_folder, exp_name, 'train')
        run_evaluation.run_evaluation_main(para_file,b_new_validation_data=True, train_dir=TRAIN_LOGDIR)

        # get miou
        get_miou_of_overall_and_class_1_step(working_dir, para_file, eval_output)

        # move eval dir for next run.
        io_function.move_file_to_dst(eval_dir,bak_miou_dir,overwrite=False)



    # save to excel file
    train_out_table_pd = pd.DataFrame(eval_output)
    with pd.ExcelWriter(output_file) as writer:
        train_out_table_pd.to_excel(writer, sheet_name='training parameter and results')
        # set format
        workbook = writer.book
        format = workbook.add_format({'num_format': '#0.000'})
        train_out_table_sheet = writer.sheets['training parameter and results']
        train_out_table_sheet.set_column('O:P',None,format)

    os.chdir(curr_dir)

def main(options, args):

    root_dir = args[0]
    folder_pattern = options.folder_pattern
    para_file = options.para_file
    output_file = options.output
    if output_file is None:
        output_file = os.path.basename(root_dir) + '.xlsx'

    run_evaluation_multi_trained_models(root_dir,folder_pattern,para_file,output_file)


if __name__ == '__main__':
    usage = "usage: %prog [options] training_root_dir "
    parser = OptionParser(usage=usage, version="1.0 2021-03-17")
    parser.description = 'Introduction: collect parameters and training results (miou) '

    parser.add_option("-p", "--para_file",
                      action="store", dest="para_file",default='main_para.ini',
                      help="the parameters file")

    parser.add_option("-o", "--output",
                      action="store", dest="output", #default="accuracy_table.xlsx",
                      help="the output file path ")

    parser.add_option("-f", "--folder_pattern",
                      action="store", dest="folder_pattern",default='multiArea_deeplabv3P_?????',
                      help="the pattern of training folder")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)
    main(options, args)
