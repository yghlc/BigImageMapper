#!/usr/bin/env python
# Filename: whole_procedure 
"""
introduction: run the whole procedure, similar to working_dir/exe.sh but in Python

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 22 February, 2021
"""

import os, sys
from optparse import OptionParser
import time

import GPUtil

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters
import basic_src.io_function as io_function
import basic_src.basic as basic

eo_dir = os.path.expanduser('~/codes/PycharmProjects/Landuse_DL')
time_txt = 'time_cost.txt'  # don't change the filename


def remove_previous_data_or_results(para_file):
    # remove previous data or result if necessary
    if os.path.isfile(time_txt):
        io_function.delete_file_or_dir(time_txt)
    command_string = os.path.join(eo_dir, 'workflow', 'remove_previous_data.py') + ' ' + para_file
    basic.os_system_exit_code(command_string)


def extract_sub_images_using_training_polygons(para_file):
    # extract sub_images based on the training polgyons
    command_string = os.path.join(eo_dir, 'workflow', 'get_sub_images_multi_regions.py') + ' ' + para_file
    basic.os_system_exit_code(command_string)


def split_sub_images(para_file):
    command_string = os.path.join(eo_dir, 'workflow', 'split_sub_images.py') + ' ' + para_file
    basic.os_system_exit_code(command_string)


def training_img_augment(para_file):
    command_string = os.path.join(eo_dir, 'workflow', 'training_img_augment.py') + ' ' + para_file
    basic.os_system_exit_code(command_string)


def split_train_val(para_file):
    command_string = os.path.join(eo_dir, 'workflow', 'split_train_val.py') + ' ' + para_file
    basic.os_system_exit_code(command_string)


def build_TFrecord_tf1x(para_file):
    command_string = os.path.join(eo_dir, 'workflow', 'build_TFrecord_tf1x.py') + ' ' + para_file
    basic.os_system_exit_code(command_string)


def training(para_file, gpu_num):
    # if gpus is not None:
    #     gpu_num = len(gpus)
    # if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
    #     gpus_str = [ item.strip() for item in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    #     if len(gpus_str) == 1 and gpus_str[0] == '':
    #         gpu_num = 1     # no gpu, set a 1
    #     else:
    #         gpu_num = len(gpus)
    # else:
    #     # find available gpus
    #     deviceIDs = GPUtil.getAvailable(order='first', limit=100, maxLoad=0.5,
    #                                     maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
    #     gpu_num = len(deviceIDs)

    command_string = os.path.join(eo_dir, 'workflow', 'deeplab_train.py') + ' ' + para_file + ' ' + str(gpu_num)
    basic.os_system_exit_code(command_string)


def export_model(para_file):
    command_string = os.path.join(eo_dir, 'workflow', 'export_graph.py') + ' ' + para_file
    basic.os_system_exit_code(command_string)


def inference(para_file):
    outdir = parameters.get_directory(para_file, 'inf_output_dir')
    if os.path.isdir(outdir):
        io_function.delete_file_or_dir(outdir)
    command_string = os.path.join(eo_dir, 'workflow', 'parallel_prediction.py') + ' ' + para_file
    basic.os_system_exit_code(command_string)


def post_processing_backup(para_file, inf_post_note=None):
    command_string = os.path.join(eo_dir, 'workflow', 'postProcess.py') + ' ' + para_file
    if inf_post_note is not None:
        command_string += ' ' + str(inf_post_note)
    basic.os_system_exit_code(command_string)

def run_whole_procedure(para_file, working_dir=None, gpus=None, gpu_num=1):
    '''
    run the whole procedure of training, prediction, and post-processing
    :param working_dir: working folder, defulat is current folder
    :param para_file: the main parameters
    :param gpus: a lists of gpu to use
    :return: True if successful
    '''
    curr_dir = os.getcwd()
    if working_dir is not None:
        os.chdir(working_dir)
    if gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(item) for item in gpus])
        gpu_num = len(gpus)

    SECONDS = time.time()

    remove_previous_data_or_results(para_file)
    extract_sub_images_using_training_polygons(para_file)

    # ## preparing training images.
    split_sub_images(para_file)
    training_img_augment(para_file)
    split_train_val(para_file)
    ## convert to TFrecord
    build_TFrecord_tf1x(para_file)

    duration = time.time() - SECONDS
    os.system('echo "$(date): time cost of preparing training data: %.2f seconds">>time_cost.txt' % duration)

    # ## training
    training(para_file, gpu_num)

    ## export model
    export_model(para_file)

    ## inference
    inference(para_file)

    # post processing and copy results, inf_post_note indicate notes for inference and post-processing
    post_processing_backup(para_file)

    #################################################
    ### conduct polygon-based change detection based on the multi-temporal mapping results
    # cd_code=~/codes/PycharmProjects/ChangeDet_DL
    # ${cd_code}/thawSlumpChangeDet/polygons_cd_multi_exe.py ${para_file} ${test_name}

    if working_dir is not None:
        os.chdir(curr_dir)

    pass


def main(options, args):
    pass


if __name__ == '__main__':
    usage = "usage: %prog [options] "
    parser = OptionParser(usage=usage, version="1.0 2021-02-22")
    parser.description = 'Introduction: run the whole procedure: training, inference, and post-processing '

    parser.add_option("-n", "--gpu_num",
                      action="store", dest="gpu_num", type=int, default=2,
                      help="the number of GPUs for training")

    (options, args) = parser.parse_args()
    # if len(sys.argv) < 2:
    #     parser.print_help()
    #     sys.exit(2)

    main(options, args)
