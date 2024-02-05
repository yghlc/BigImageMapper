#!/usr/bin/env python
# Filename: clip_train.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 24 January, 2024
"""

import os,sys
from optparse import OptionParser
import time
from datetime import datetime
import torch
import numpy as np

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters
import basic_src.io_function as io_function

from prediction_clip import prepare_dataset, run_prediction

import clip

def prepare_training_data(WORK_DIR, para_file, transform, test=False):

    training_regions = parameters.get_string_list_parameters_None_if_absence(para_file,'training_regions')
    if training_regions is None or len(training_regions) < 1:
        raise ValueError('No training area is set in %s'%para_file)

    # TODO: support multiple training regions
    area_ini = training_regions[0]
    area_name = parameters.get_string_parameters(area_ini, 'area_name')
    area_remark = parameters.get_string_parameters(area_ini, 'area_remark')
    area_time = parameters.get_string_parameters(area_ini, 'area_time')

    area_name_remark_time = area_name + '_' + area_remark + '_' + area_time
    area_save_dir = os.path.join(WORK_DIR, area_name_remark_time)

    # prepare training data
    train_image_dir = parameters.get_directory(area_ini, 'input_image_dir')
    train_image_or_pattern = parameters.get_string_parameters(area_ini, 'input_image_or_pattern')
    # TODO need to check preprocess, do we need to define it?
    in_dataset = prepare_dataset(para_file,area_ini,area_save_dir, train_image_dir, train_image_or_pattern,
                                 transform, test=test)
    return in_dataset

def generate_pseudo_labes(dataset, data_loader, save_dir, device, model, clip_prompt,
                          probs_thr=0.6, topk=10, version=1):
    # run prediction
    predict_probs, ground_truths = run_prediction(model,data_loader, clip_prompt, device)

    # save the results, as training data
    classes = dataset.classes
    save_str_list = []
    for c, name in enumerate(classes):
        pre_probs_per_class = predict_probs[:, c]
        indices = np.argsort(-pre_probs_per_class)[:topk]
        for ind in indices:
            im, label, im_path = dataset[ind]
            save_str_list.append(im_path + ' ' + str(c) + ' ' + str(label))

    save_path_txt = os.path.join(save_dir,'pseudo_v{}_train_{}shot.txt'.format(version,topk))
    io_function.save_list_to_txt(save_path_txt, save_str_list)


def training_zero_shot(para_file, network_ini, WORK_DIR, train_save_dir, device, model, preprocess):
    # without any human input training data
    train_dataset = prepare_training_data(WORK_DIR, para_file, preprocess, test=True)

    num_workers = parameters.get_digit_parameters(para_file,'process_num','int')
    train_batch_size = parameters.get_digit_parameters(network_ini,'batch_size','int')

    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    # get training label
    clip_prompt = parameters.get_string_parameters(para_file, 'clip_prompt')
    generate_pseudo_labes(train_dataset, data_loader,train_save_dir, device, model,clip_prompt)



def training_few_shot():
    # with a few human input training data
    pass


def train_clip(WORK_DIR, para_file, gpu_num):
    network_ini = parameters.get_string_parameters(para_file, 'network_setting_ini')
    expr_name = parameters.get_string_parameters(para_file, 'expr_name')

    train_save_dir = os.path.join(WORK_DIR, expr_name)
    if os.path.isdir(train_save_dir) is False:
        io_function.mkdir(train_save_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = parameters.get_string_parameters(network_ini, 'model_type')
    model, preprocess = clip.load(model_type, device=device)

    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)


    b_a_few_shot_training = parameters.get_bool_parameters(para_file,'a_few_shot_training')
    if b_a_few_shot_training:
        training_few_shot()
    else:
        training_zero_shot(para_file, network_ini, WORK_DIR, train_save_dir, device, model, preprocess )

def clip_train_main(para_file,gpu_num=1):
    print(datetime.now(),"train CLIP")
    SECONDS = time.time()

    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    WORK_DIR = os.getcwd()
    train_clip(WORK_DIR, para_file, gpu_num)

    duration = time.time() - SECONDS
    os.system('echo "$(date): time cost of training: %.2f seconds">>time_cost.txt' % duration)


def main(options, args):

    para_file = args[0]
    clip_train_main(para_file)


if __name__ == '__main__':
    usage = "usage: %prog [options] para_file"
    parser = OptionParser(usage=usage, version="1.0 2024-01-24")
    parser.description = 'Introduction: fine-tune the clip model using custom data'

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)