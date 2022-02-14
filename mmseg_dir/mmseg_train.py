#!/usr/bin/env python
# Filename: mmseg_train.py 
"""
introduction: run training of a deep learning model for semantic segmentation using MMSegmentation

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 11 January, 2022
"""

import os,sys
import os.path as osp
from optparse import OptionParser
from datetime import datetime
import time

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters
import basic_src.io_function as io_function

from workflow.deeplab_train import get_iteration_num

# the poython with mmseg and pytorch installed
open_mmlab_python = 'python'

# open-mmlab models
from mmcv.utils import Config


# to fix: Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library.
#        Try to import numpy first or set the threading layer accordingly. Set MKL_SERVICE_FORCE_INTEL to force it.
os.environ['MKL_THREADING_LAYER'] = 'GNU'


def train_evaluation_mmseg(WORK_DIR,mmseg_repo_dir,config_file, expr_name, para_file, network_setting_ini, gpu_num):
    '''
    run the training process of MMSegmentation
    :param WORK_DIR:
    :param mmseg_repo_dir:
    :param config_file:
    :param expr_name:
    :param para_file:
    :param network_setting_ini:
    :param gpu_num:
    :return:
    '''
    dist_train_sh = osp.join(mmseg_repo_dir,'tools','dist_train.sh')

    command_string = dist_train_sh + ' ' \
                    + config_file + ' ' \
                    + str(gpu_num)

    res = os.system(command_string)
    if res != 0:
        sys.exit(1) # sometime the res is 256 and bash cannot recognize that, then continue run.

    return True


def modify_dataset(cfg,para_file,network_setting_ini,gpu_num):
    datetype = 'RSImagePatches'
    cfg.dataset_type = datetype
    cfg.data_root = './'

    ## There are still two more crop_size in pipleline, after choosing the base_config_file, we already choose the crop size

    # image_crop_size = parameters.get_string_list_parameters(para_file, 'image_crop_size')
    # image_crop_size = [ int(item) for item in image_crop_size]
    # if len(image_crop_size) != 2 and image_crop_size[0].isdigit() and image_crop_size[1].isdigit():
    #     raise ValueError('image_crop_size should be height,width')
    # cfg.crop_size = (image_crop_size[0],image_crop_size[1])

    training_sample_list_txt = parameters.get_string_parameters(para_file,'training_sample_list_txt')
    validation_sample_list_txt = parameters.get_string_parameters(para_file,'validation_sample_list_txt')

    split_list = ['train','val','test']
    for split in split_list:
        # dataset in train
        cfg.data[split]['type'] = datetype
        cfg.data[split]['data_root'] = './'
        cfg.data[split]['img_dir'] = 'split_images'
        cfg.data[split]['ann_dir'] = 'split_labels'
        if split=='train':
            cfg.data[split]['split'] = [osp.join('list',training_sample_list_txt)]
        else:
            # set val and test to validation, when run real test (prediction) for entire RS images, we will set test again.
            cfg.data[split]['split'] = [osp.join('list', validation_sample_list_txt)]

    # set None for test
    cfg.data['test']['img_dir'] = None
    cfg.data['test']['ann_dir'] = None
    cfg.data['test']['split'] = None


    # setting based on batch size
    batch_size = parameters.get_digit_parameters(network_setting_ini,'batch_size','int')
    if batch_size % gpu_num != 0:
        raise ValueError('Batch size (%d) cannot be divided by gpu num (%d)'%(batch_size,gpu_num))

    cfg.data['samples_per_gpu'] = int(batch_size/gpu_num)
    cfg.data['workers_per_gpu'] = int(batch_size/gpu_num) + 2 # set worker a litter higher to utilize CPU

    return True

def updated_config_file(WORK_DIR, expr_name,base_config_file,save_path,para_file,network_setting_ini,gpu_num):
    '''
    update mmseg config file and save to a new file in local working folder
    :param WORK_DIR:
    :param expr_name:
    :param base_config_file:
    :param save_path:
    :param para_file:
    :param network_setting_ini:
    :return:
    '''
    # read the config, then modifi some, them dump to disk
    cfg = Config.fromfile(base_config_file)
    # 4 basic component: dataset, model, schedule, default_runtime
    # change model (no need): when we choose a base_config_file, we already choose a model, including backbone)

    # change dataset
    modify_dataset(cfg,para_file,network_setting_ini,gpu_num)

    # change schedule
    iteration_num = get_iteration_num(WORK_DIR,para_file,network_setting_ini)
    cfg.runner['max_iters'] = iteration_num

    checkpoint_interval = parameters.get_digit_parameters(network_setting_ini,'checkpoint_interval','int')
    cfg.checkpoint_config['interval'] = checkpoint_interval
    evaluation_interval = parameters.get_digit_parameters(network_setting_ini,'evaluation_interval','int')
    cfg.evaluation['interval'] = evaluation_interval


    # change runtime (log level, resume_from or load_from)
    cfg.work_dir = os.path.join(WORK_DIR,expr_name)

    # update parameters for testing (used in later step of prediction)
    updated_config_file_for_test(cfg)

    # dump config
    cfg.dump(save_path)
    return True


def updated_config_file_for_test(cfg):
    loadimg = 'LoadRSImagePatch'
    cfg.test_pipeline[0] = dict(type=loadimg)
    cfg.data['test']['pipeline'][0] = dict(type=loadimg)


def mmseg_train_main(para_file,gpu_num):
    print(datetime.now(),"train MMSegmentation")
    SECONDS = time.time()

    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in the current folder: %s' % (para_file, os.getcwd()))

    network_setting_ini = parameters.get_string_parameters(para_file, 'network_setting_ini')
    mmseg_repo_dir = parameters.get_directory(network_setting_ini, 'mmseg_repo_dir')
    mmseg_config_dir = osp.join(mmseg_repo_dir,'configs')
    if os.path.isdir(mmseg_config_dir) is False:
        raise ValueError('%s does not exist' % mmseg_config_dir)

    base_config_file = parameters.get_string_parameters(network_setting_ini, 'base_config')
    base_config_file = os.path.join(mmseg_config_dir,base_config_file)
    if os.path.isfile(base_config_file) is False:
        raise IOError('%s does not exist'%base_config_file)

    global open_mmlab_python
    open_mmlab_python = parameters.get_file_path_parameters(network_setting_ini, 'open-mmlab-python')

    WORK_DIR = os.getcwd()
    expr_name = parameters.get_string_parameters(para_file, 'expr_name')

    # copy the base_config_file, then save to to a new one
    config_file = osp.join(WORK_DIR,osp.basename(io_function.get_name_by_adding_tail(base_config_file,expr_name))) 
    if updated_config_file(WORK_DIR, expr_name,base_config_file,config_file,para_file,network_setting_ini,gpu_num) is False:
        raise ValueError('Getting the config file failed')

    train_evaluation_mmseg(WORK_DIR,mmseg_repo_dir,config_file, expr_name, para_file, network_setting_ini, gpu_num)

    duration = time.time() - SECONDS
    os.system('echo "$(date): time cost of training: %.2f seconds">>time_cost.txt' % duration)


def main(options, args):
    para_file = args[0]
    gpu_num = int(args[1])
    mmseg_train_main(para_file,gpu_num)

if __name__ == '__main__':
    usage = "usage: %prog [options] para_file gpu_num"
    parser = OptionParser(usage=usage, version="1.0 2022-01-11")
    parser.description = 'Introduction: training and evaluating of Model available on MMSegmentation '

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)