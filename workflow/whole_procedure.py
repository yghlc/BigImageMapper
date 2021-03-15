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
    # command_string = os.path.join(eo_dir, 'workflow', 'remove_previous_data.py') + ' ' + para_file
    # basic.os_system_exit_code(command_string)
    from remove_previous_data import remove_previous_data
    return remove_previous_data(para_file)


def extract_sub_images_using_training_polygons(para_file):
    trainImg_dir = parameters.get_string_parameters(para_file, 'input_train_dir')
    labelImg_dir = parameters.get_string_parameters(para_file, 'input_label_dir')
    if os.path.isdir(trainImg_dir)  and os.path.isdir(labelImg_dir):
        basic.outputlogMessage('warning, sub-image and sub-label folder exists, skip extracting sub-images')
        return
    # extract sub_images based on the training polgyons
    # command_string = os.path.join(eo_dir, 'workflow', 'get_sub_images_multi_regions.py') + ' ' + para_file
    # basic.os_system_exit_code(command_string)
    from get_sub_images_multi_regions import get_sub_images_multi_regions
    return get_sub_images_multi_regions(para_file)


def split_sub_images(para_file):
    if os.path.isdir('split_images') and os.path.isdir('split_labels'):
        basic.outputlogMessage('warning, split_image sand split_labels folder exists, skip splitting sub-images')
        return
    # command_string = os.path.join(eo_dir, 'workflow', 'split_sub_images.py') + ' ' + para_file
    # basic.os_system_exit_code(command_string)
    from split_sub_images import split_sub_images
    return split_sub_images(para_file)


def training_img_augment(para_file):
    if os.path.isfile(os.path.join('list', 'images_including_aug.txt')):
        basic.outputlogMessage('warning, list/images_including_aug.txt exists, skip data augmentation')
        return
    # command_string = os.path.join(eo_dir, 'workflow', 'training_img_augment.py') + ' ' + para_file
    # basic.os_system_exit_code(command_string)
    from training_img_augment import training_img_augment
    return training_img_augment(para_file)


def split_train_val(para_file):

    train_sample_txt = parameters.get_string_parameters(para_file, 'training_sample_list_txt')
    val_sample_txt = parameters.get_string_parameters(para_file, 'validation_sample_list_txt')
    train_list_txt = os.path.join('list', train_sample_txt)
    val_list_txt = os.path.join('list', val_sample_txt)
    if os.path.isfile(train_list_txt) and os.path.isfile(val_list_txt):
        basic.outputlogMessage('warning, split sample list exists, skip split_train_val')
        return
    # command_string = os.path.join(eo_dir, 'workflow', 'split_train_val.py') + ' ' + para_file
    # basic.os_system_exit_code(command_string)
    import split_train_val
    return split_train_val.split_train_val(para_file)


def build_TFrecord_tf1x(para_file):
    if os.path.isdir('tfrecord'):
        basic.outputlogMessage('warning, tfrecord exists, skip build_TFrecord_tf1x')
        return
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

    # the script can check the trained iteration and decide to train or not
    # command_string = os.path.join(eo_dir, 'workflow', 'deeplab_train.py') + ' ' + para_file + ' ' + str(gpu_num)
    # basic.os_system_exit_code(command_string)
    import deeplab_train
    deeplab_train.deeplab_train_main(para_file, gpu_num)


def export_model(para_file):
    # the script can check the trained model and decide to export or not
    command_string = os.path.join(eo_dir, 'workflow', 'export_graph.py') + ' ' + para_file
    basic.os_system_exit_code(command_string)


def inference(para_file):
    outdir = parameters.get_directory(para_file, 'inf_output_dir')
    # don't remove it automatically
    # if os.path.isdir(outdir):
    #     io_function.delete_file_or_dir(outdir)
    # the script will check whether each image has been predicted
    command_string = os.path.join(eo_dir, 'workflow', 'parallel_prediction.py') + ' ' + para_file
    basic.os_system_exit_code(command_string)


def post_processing_backup(para_file, inf_post_note=None, b_skip_getshp=False, test_id=None):
    # the script will check whether each image has been post-processed
    # command_string = os.path.join(eo_dir, 'workflow', 'postProcess.py') + ' ' + para_file
    # if inf_post_note is not None:
    #     command_string += ' ' + str(inf_post_note)
    # basic.os_system_exit_code(command_string)
    import postProcess
    postProcess.postProcess(para_file,inf_post_note, b_skip_getshp=b_skip_getshp,test_id=test_id)

def run_whole_procedure(para_file, working_dir=None, gpus=None, gpu_num=1, b_train_only=False):
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
    elif 'CUDA_VISIBLE_DEVICES' in os.environ.keys() and len(os.environ['CUDA_VISIBLE_DEVICES'])>0:
        gpus_str = [ item.strip() for item in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        gpu_num = len(gpus_str)
    else:
        # print('gpu number is set to 1')
        # gpu_num = 1
        pass



    SECONDS = time.time()

    # remove_previous_data_or_results(para_file)    # don't automatic remove data
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

    if b_train_only is False:
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
    para_file = args[0]
    gpu_num = options.gpu_num
    b_train_only = options.b_train_only

    run_whole_procedure(para_file,b_train_only=b_train_only,gpu_num=gpu_num)


if __name__ == '__main__':
    usage = "usage: %prog [options] main_para.ini"
    parser = OptionParser(usage=usage, version="1.0 2021-02-22")
    parser.description = 'Introduction: run the whole procedure: training, inference, and post-processing '

    # should use CUDA_VISIBLE_DEVICES to set GPUs
    # still need to set gpu_num if CUDA_VISIBLE_DEVICES is not set sometime
    parser.add_option("-n", "--gpu_num",
                      action="store", dest="gpu_num", type=int, default=2,
                      help="the number of GPUs for training")

    parser.add_option("-t", "--b_train_only",
                      action="store_true", dest="b_train_only",default=False,
                      help="indicate to run training only, not ")


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)
