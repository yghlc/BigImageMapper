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

# the poython with mmseg and pytorch installed
open_mmlab_python = 'python'

# open-mmlab models
from mmcv.utils import Config


def train_evaluation_mmseg(WORK_DIR,config_file, expr_name, para_file, network_setting_ini, gpu_num):
    '''
    run the training process of MMSegmentation
    :param WORK_DIR:
    :param config_file:
    :param expr_name:
    :param para_file:
    :param network_setting_ini:
    :param gpu_num:
    :return:
    '''

    pass

def updated_config_file(WORK_DIR, expr_name,base_config_file,save_path,para_file,network_setting_ini):
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
    io_function.copy_file_to_dst(base_config_file,save_path,overwrite=False)
    cfg = Config.fromfile(args.config)
    # 4 basic component: dataset, model, schedule, default_runtime
    # change model (no need): when we choose a base_config_file, we already choose a model, including backbone)

    # change dataset
    cfg.dataset_type = 'RSImagePatches'

    # change schedule

    # change runtime (log level, resume_from or load_from)
    cfg.work_dir = os.path.join(WORK_DIR,expr_name)

    # dump config
    cfg.dump(save_path)
    return True



def mmseg_train_main(para_file,gpu_num):
    print(datetime.now(),"train MMSegmentation")
    SECONDS = time.time()

    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in the current folder: %s' % (para_file, os.getcwd()))

    network_setting_ini = parameters.get_string_parameters(para_file, 'network_setting_ini')
    mmseg_config_dir = parameters.get_directory(network_setting_ini, 'mmseg_config_dir')
    if os.path.isdir(mmseg_config_dir) is False:
        raise ValueError('%s does not exist' % mmseg_config_dir)

    base_config_file = parameters.get_string_parameters(para_file, 'base_config')
    base_config_file = os.path.join(mmseg_config_dir,base_config_file)
    if os.path.isfile(base_config_file) is False:
        raise IOError('%s does not exist'%base_config_file)

    global open_mmlab_python
    open_mmlab_python = parameters.get_file_path_parameters(network_setting_ini, 'open-mmlab-python')

    WORK_DIR = os.getcwd()
    expr_name = parameters.get_string_parameters(para_file, 'expr_name')

    # copy the base_config_file, then save to to a new one
    config_file = io_function.get_name_by_adding_tail(base_config_file,expr_name)
    if updated_config_file(WORK_DIR, expr_name,base_config_file,config_file,para_file,network_setting_ini) is False:
        raise ValueError('Getting the config file failed')

    train_evaluation_mmseg(WORK_DIR,config_file, expr_name, para_file, network_setting_ini, gpu_num)

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