#!/usr/bin/env python
# Filename: plot_mmseg_loss_miou.py
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 04 February, 2022
"""

import os, sys
from optparse import OptionParser
import json

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)

import basic_src.io_function as io_function
import basic_src.basic as basic

import matplotlib
# must be before importing matplotlib.pyplot or pylab!
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_mmseg_log_json(json_path,miou_class_0,miou_class_1):

    log_dict_train = {'epoch':[], 'iter':[],'lr':[],'memory':[],'loss':[],'data_time':[],'time':[]}

    log_dict_val = {'epoch':[], 'iter':[],'lr':[],miou_class_0:[],miou_class_1:[],'0_mIoU':[]}
    with open(json_path, 'r') as log_file:
        previous_line_log = None
        for line in log_file:
            log = json.loads(line.strip())
            # skip lines without `epoch` field
            if 'epoch' not in log:
                continue
            epoch = log.pop('epoch')
            # print(log)
            # print(epoch)
            # print(log.keys())
            if log['mode'] == 'train':
                log_dict_train['epoch'].append(epoch)
                log_dict_train['iter'].append(log['iter'])
                log_dict_train['lr'].append(log['lr'])
                log_dict_train['memory'].append(log['memory'])
                log_dict_train['loss'].append(log['loss'])
                log_dict_train['data_time'].append(log['data_time'])
                log_dict_train['time'].append(log['time'])
            elif log['mode'] == 'val':
                log_dict_val['epoch'].append(epoch)
                # log_dict_val['iter'].append(log['iter'])    #
                log_dict_val['iter'].append(previous_line_log['iter'])      # we should use the iter in previous line (train), after that training, it run valdation
                log_dict_val['lr'].append(log['lr'])
                log_dict_val[miou_class_0].append(log[miou_class_0])
                log_dict_val[miou_class_1].append(log[miou_class_1])
                log_dict_val['0_mIoU'].append(log['0_mIoU'])
            else:
                raise ValueError('unknown model: %s'%log['mode'])

            previous_line_log = log

    return log_dict_train, log_dict_val


def plot_loss_vs_epoch_iter(log_dict_train,save_path):
    '''
    plot loss vs epoch and iteration in one figure
    :param log_dict_train:
    :param save_path:
    :return:
    '''

    # plot the histogram
    fig = plt.figure(figsize=(6,4)) #
    ax1 = fig.add_subplot(111)

    line_style = ['b-', 'g-', 'r-', 'r-.']

    # plot loss
    ax1.plot(log_dict_train['iter'], log_dict_train['loss'], 'k-.', label='loss', linewidth=0.8)
    ax1.set_xlim([0, max(log_dict_train['iter'])])
    ax1.set_ylim([0,1.0])
    ax1.set_xlabel('iteration', fontsize=10)

    # plot miou
    ax2 = ax1.twiny()
    min_epoch, max_epoch = min(log_dict_train['epoch']), max(log_dict_train['epoch'])
    print(min_epoch, max_epoch)
    # print(ax1.get_xlim())
    ax2.set_xlim([min_epoch, max_epoch])
    # print(labels)
    # ax2.set_xticklabels(labels)
    ax2.set_xlabel("epoch", fontsize=10)  # color="red",
    ax1.legend(fontsize=10, loc="upper right")  # loc="upper left"


    plt.tight_layout()  # adjust the layout, avoid cutoff some label to title
    # plt.show()
    plt.savefig(save_path, dpi=200)  # 300
    basic.outputlogMessage('save to %s'%save_path)


def plot_miou_vs_epoch_iter(log_dict_val,miou_key,save_path):
    '''
    plot miou vs epoch and iteration
    :param log_dict_val:
    :param miou_key:
    :param save_path:
    :return:
    '''
    # plot the histogram
    fig = plt.figure(figsize=(6,4)) #
    ax1 = fig.add_subplot(111)

    line_style = ['b-', 'g-', 'r-', 'r-.']

    # plot loss
    ax1.plot(log_dict_val['iter'], log_dict_val[miou_key], 'r-.', label=miou_key, linewidth=0.8)
    ax1.set_xlim([0, max(log_dict_val['iter'])])
    ax1.set_ylim([0, 1.0])
    ax1.set_xlabel('iteration', fontsize=10)

    # plot miou
    ax2 = ax1.twiny()
    min_epoch, max_epoch = min(log_dict_val['epoch']), max(log_dict_val['epoch'])
    print(min_epoch, max_epoch)
    # print(ax1.get_xlim())
    ax2.set_xlim([min_epoch, max_epoch])
    # print(labels)
    # ax2.set_xticklabels(labels)
    ax2.set_xlabel("epoch", fontsize=10)  # color="red",
    ax1.legend(fontsize=10, loc="lower right")  # loc="upper left"


    plt.tight_layout()  # adjust the layout, avoid cutoff some label to title
    # plt.show()
    plt.savefig(save_path, dpi=200)  # 300
    basic.outputlogMessage('save to %s'%save_path)


def plot_mmseg_loss_miou_acc_main(json_path, save_file_pre=None):

    if os.path.isfile(json_path) is False:
        return False
    if save_file_pre is None:
        file_name = os.path.splitext(os.path.basename(json_path))[0]
    else:
        file_name = save_file_pre

    # TODO: background and thawslump should be read from the setting file
    miou_class_0 = '0_IoU.%s'%'background'
    miou_class_1 = '0_IoU.%s'%'thawslump'

    save_dir = os.path.dirname(json_path)
    save_path = os.path.join(save_dir, file_name + '.jpg')

    # log_dict_train = {'epoch':[], 'iter':[],'lr':[],'memory':[],'loss':[],'data_time':[],'time':[]}
    # log_dict_val = {'epoch':[], 'iter':[],'lr':[],miou_class_0:[],miou_class_1:[],'0_mIoU':[]}

    log_dict_train, log_dict_val = load_mmseg_log_json(json_path,miou_class_0,miou_class_1)
    # print(log_dict_train, log_dict_val)
    # io_function.save_dict_to_txt_json('log_dict_train.txt',log_dict_train)
    # io_function.save_dict_to_txt_json('log_dict_val.txt',log_dict_val)
    plot_loss_vs_epoch_iter(log_dict_train,io_function.get_name_by_adding_tail(save_path,'loss'))

    plot_miou_vs_epoch_iter(log_dict_val,miou_class_1,io_function.get_name_by_adding_tail(save_path,'miou'))


def test_plot_mmseg_loss_miou_acc_main():
    json_path = os.path.expanduser('~/Data/tmp_data/test_mmsegmentation/test_landuse_dl/exp19/20220203_101247.log.json')
    plot_mmseg_loss_miou_acc_main(json_path, save_file_pre=None)


def main(options, args):
    json_path = args[0]
    # test_plot_mmseg_loss_miou_acc_main()
    plot_mmseg_loss_miou_acc_main(json_path, save_file_pre=options.save_file_pre)


if __name__ == '__main__':
    usage = "usage: %prog [options] json_path  "
    parser = OptionParser(usage=usage, version="1.0 2022-02-04")
    parser.description = 'Introduction:  plot curve of training loss and miou from mmseg log file.'

    parser.add_option("-s", "--save_file_pre",
                      action="store", dest="save_file_pre",
                      help="the prefix for saving files")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)
    main(options, args)