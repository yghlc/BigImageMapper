#!/usr/bin/env python
# Filename: fine_tune_sam.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 06 April, 2024
"""

import os, sys
import time
from datetime import datetime
from optparse import OptionParser
import torch

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters
import basic_src.io_function as io_function


def fine_tune_sam(WORK_DIR, para_file, pre_train_model='', gpu_num=1):
    network_ini = parameters.get_string_parameters(para_file, 'network_setting_ini')
    expr_name = parameters.get_string_parameters(para_file, 'expr_name')

    train_save_dir = os.path.join(WORK_DIR, expr_name)
    if os.path.isdir(train_save_dir) is False:
        io_function.mkdir(train_save_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = parameters.get_file_path_parameters(network_ini, 'checkpoint')
    model_type = parameters.get_string_parameters(network_ini, 'model_type')

    # get training data


    # train SAM


    # save trained models





def fine_tune_sam_main(para_file, pre_train_model='',gpu_num=1):
    print(datetime.now(), "fine-tune Segment anything models")
    SECONDS = time.time()

    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    WORK_DIR = os.getcwd()
    fine_tune_sam(WORK_DIR, para_file, pre_train_model=pre_train_model, gpu_num=gpu_num)

    duration = time.time() - SECONDS
    os.system('echo "$(date): time cost for training SAM: %.2f seconds">>time_cost.txt' % duration)

def main(options, args):

    para_file = args[0]
    pre_train_model = options.pretrain_model

    fine_tune_sam_main(para_file, pre_train_model=pre_train_model)

if __name__ == '__main__':
    usage = "usage: %prog [options] para_file"
    parser = OptionParser(usage=usage, version="1.0 2024-04-06")
    parser.description = 'Introduction: fine-tuning segment anything models '

    parser.add_option("-m", "--pretrain_model",
                      action="store", dest="pretrain_model",default='',
                      help="the pre-trained model")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)
