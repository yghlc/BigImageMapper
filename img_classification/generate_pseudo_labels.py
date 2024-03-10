#!/usr/bin/env python
# Filename: generate_pseudo_labels.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 10 March, 2024
"""

import os,sys
import numpy as np
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import basic_src.io_function as io_function
import parameters

from prediction_clip import prepare_dataset, run_prediction, calculate_top_k_accuracy
from train_clip import prepare_training_data, log_string

import torch
import clip

def generate_pseudo_labels(dataset, data_loader, save_dir, device, model, clip_prompt,
                          probs_thr=0.6, topk=10, version=1):
    # run prediction
    predict_probs, ground_truths = run_prediction(model,data_loader, clip_prompt, device)

    # save the results, as training data
    classes = dataset.classes
    save_str_list = []
    for c, name in enumerate(classes):
        pre_probs_per_class = predict_probs[:, c].cpu()
        indices = np.argsort(-pre_probs_per_class)[:topk]
        for ind in indices:
            im, label, im_path = dataset[ind]
            save_str_list.append(im_path + ' ' + str(c) + ' ' + str(label) + ' ' + '%f'%float(pre_probs_per_class[ind].cpu()))

    save_path_txt = os.path.join(save_dir,'pseudo_v{}_train_{}shot.txt'.format(version,topk))
    io_function.save_list_to_txt(save_path_txt, save_str_list)

    return save_path_txt

def generate_pseudo_labels_main(para_file, trained_model = None, v_num=1, topk=10):

    WORK_DIR = os.getcwd()
    network_ini = parameters.get_string_parameters(para_file, 'network_setting_ini')
    expr_name = parameters.get_string_parameters(para_file, 'expr_name')

    train_save_dir = os.path.join(WORK_DIR, expr_name)
    if os.path.isdir(train_save_dir) is False:
        io_function.mkdir(train_save_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = parameters.get_string_parameters(network_ini, 'model_type')
    model, preprocess = clip.load(model_type, device=device)

    dataset = prepare_training_data(WORK_DIR, para_file, preprocess, test=True)

    num_workers = parameters.get_digit_parameters(para_file, 'process_num', 'int')
    train_batch_size = parameters.get_digit_parameters(network_ini, 'batch_size', 'int')

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    probability_threshold = parameters.get_digit_parameters(network_ini, 'probability_threshold', 'float')

    if trained_model is not None:
        log_string("Loading pretrained model : [%s]" % trained_model)
        checkpoint = torch.load(open(trained_model, 'rb'), map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])

    # get pseudo labels
    clip_prompt = parameters.get_string_parameters(para_file, 'clip_prompt')
    training_samples_txt = generate_pseudo_labels(dataset, data_loader, train_save_dir, device, model, clip_prompt,
                                                      probs_thr=probability_threshold, topk=topk, version=v_num)

    log_string('saved pseudo labels to %s'%training_samples_txt)


def main(options, args):

    para_file = args[0]
    v_num = args[1]
    topk = args[2]
    generate_pseudo_labels_main(para_file, v_num, topk)


if __name__ == '__main__':
    usage = "usage: %prog [options] para_file v_num topk"
    parser = OptionParser(usage=usage, version="1.0 2024-01-24")
    parser.description = 'Introduction: run prediction and generate pseudo labels'

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)