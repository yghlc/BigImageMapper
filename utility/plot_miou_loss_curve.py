#!/usr/bin/env python
# Filename: plot_miou_loss_curve.py 
"""
introduction: plot curve of training loss, learning, and miou.

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 12 February, 2021
"""

import os, sys
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)

import basic_src.io_function as io_function
import basic_src.basic as basic

import matplotlib
# must be before importing matplotlib.pyplot or pylab!

# if os.name == 'posix' and "DISPLAY" not in os.environ:
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# wall_time_to_relative time
def wall_time_to_relative_time(wall_time_list):
    diff_hours = [(wall_time_list[idx+1] - wall_time_list[idx])/3600 for idx in range(len(wall_time_list) - 1)]
    mean_diff = sum(diff_hours)/len(diff_hours)

    relative_time = [0,mean_diff]
    acc_time = mean_diff
    for idx in range(len(diff_hours)):
        acc_time += diff_hours[idx]
        relative_time.append(acc_time)

    # print(relative_time)
    # return ["%.2f" % item for item in relative_time]
    return min(relative_time), max(relative_time)

def plot_miou_step_time(miou_dict, save_path,train_count, val_count,batch_size):

    # class_0
    # class_1
    # overall
    # step
    # wall_time

    # plot the histogram
    fig = plt.figure(figsize=(6,4)) #
    ax1 = fig.add_subplot(111)

    line_style = ['b-', 'g-', 'r-', 'r-.']

    for c_num in range(0, 1000):
        class_name = 'class_%d'%c_num
        if class_name in miou_dict.keys():
            # if "nan" value in this list, then skip
            if np.isnan(miou_dict[class_name]).any():
                print('warning, NaN value in of %d class, skip'%c_num)
                continue
            ax1.plot(miou_dict['step'], miou_dict[class_name], line_style[c_num%4], label="Class %d" % c_num, linewidth=0.8)
    ax1.plot(miou_dict['step'], miou_dict['overall'], 'k-.', label="Overall", linewidth=0.8)
    ax1.set_xlim([0, max(miou_dict['step'])])


    ax2 = ax1.twiny()    #have another x-axis for time
    ax2_tick_locations = np.array(miou_dict['wall_time'])
    # ax2.set_xlim(ax1.get_xlim())
    # ax2.set_xticks(ax2_tick_locations)
    # labels = wall_time_to_relative_time(ax2_tick_locations)
    min_t, max_t = wall_time_to_relative_time(miou_dict['wall_time'])
    print(min_t, max_t)
    # print(ax1.get_xlim())
    ax2.set_xlim([min_t, max_t])
    # print(labels)
    # ax2.set_xticklabels(labels)
    ax2.set_xlabel("Relative training time (hours)", fontsize=10)  # color="red",

    # ax2.spines['bottom'].set_color('blue')
    # ax1.spines['top'].set_color('red')
    # ax2.xaxis.label.set_color('blue')

    # plt.grid(axis='y', alpha=0.75)
    ax1.grid(axis='y', alpha=0.75)

    # ax1.legend(fontsize=10, loc="best")  # loc="upper left"
    ax1.legend(fontsize=10, loc="lower right")  # loc="upper left"
    if train_count is not None and val_count is not None and batch_size is not None:
        ax1.set_xlabel('Training iteration (Count of train & val: %d & %d, batch_size=%d)'%(train_count,val_count,batch_size))
    else:
        ax1.set_xlabel('Training iteration')
    ax1.set_ylabel('mIOU')

    plt.tight_layout()  # adjust the layout, avoid cutoff some label to title

    plt.savefig(save_path, dpi=200)  # 300
    basic.outputlogMessage('save to %s'%save_path)


def plot_loss_learnRate_step_time(loss_dict,save_path,train_count, val_count,batch_size):

    # ['total_loss', 'learning_rate', 'step', 'wall_time']
    # plot the histogram
    fig = plt.figure(figsize=(6,4)) #
    ax1 = fig.add_subplot(111)

    line_style = ['b-', 'g-', 'r-', 'r-.']

    # plot loss
    loss_line = ax1.plot(loss_dict['step'], loss_dict['total_loss'], line_style[0], label="Total loss", linewidth=0.8)
    ax1.set_xlim([0, max(loss_dict['step'])])
    if train_count is not None and val_count is not None and batch_size is not None:
        ax1.set_xlabel('Training iteration (Count of train & val: %d & %d, batch_size=%d)'%(train_count,val_count,batch_size))
    else:
        ax1.set_xlabel('Training iteration')
    ax1.set_ylabel('Total loss')

    ax2 = ax1.twiny()    #have another x-axis for time
    min_t, max_t = wall_time_to_relative_time(loss_dict['wall_time'])
    ax2.set_xlim([min_t, max_t])
    ax2.set_xlabel("Relative training time (hours)", fontsize=10)  # color="red",

    # plot learning rate
    ax3 = ax1.twinx()
    lr_line = ax3.plot(loss_dict['step'], loss_dict['learning_rate'], 'r-.', label="Learning rate", linewidth=0.5)
    ax3.set_ylabel('Learning rate')


    ax1.legend(fontsize=10, loc="upper right")  # loc="upper right"
    ax3.legend(fontsize=10, loc="upper right", bbox_to_anchor=(1.0, 0.9))  # loc="upper left"

    plt.tight_layout() # adjust the layout, avoid cutoff some label to title

    plt.savefig(save_path, dpi=200)  # 300
    basic.outputlogMessage('save to %s' % save_path)

def plot_miou_loss_main(txt_path, save_file_pre=None, train_count=None, val_count=None,batch_size=None):
    '''
    plot miou or loss curve
    :param txt_path:
    :param save_file_pre:
    :param train_count: the number of training samples
    :param val_count:  the number of validation samples
    :param batch_size: t
    :return:
    '''
    if os.path.isfile(txt_path) is False:
        return False
    if save_file_pre is None:
        file_name = os.path.splitext(os.path.basename(txt_path))[0]
    else:
        file_name = save_file_pre

    save_dir = os.path.dirname(txt_path)
    dict_data = io_function.read_dict_from_txt_json(txt_path)
    # print(dict_data)
    # for key in dict_data.keys():
    #     print(key)
    save_path = os.path.join(save_dir, file_name + '.jpg')
    if 'miou' in file_name:
        plot_miou_step_time(dict_data, save_path, train_count, val_count,batch_size)
    elif 'loss' in file_name:
        plot_loss_learnRate_step_time(dict_data, save_path,train_count, val_count,batch_size)
    else:
        raise ValueError('Cannot recognize the file name of miou of loss: %s'%os.path.basename(txt_path))

    return save_path

def main(options, args):
    txt_path = args[0]

    plot_miou_loss_main(txt_path, options.save_file_pre)



if __name__ == '__main__':

    usage = "usage: %prog [options] txt_path  "
    parser = OptionParser(usage=usage, version="1.0 2021-02-12")
    parser.description = 'Introduction:  plot curve of training loss, learning, and miou.'


    parser.add_option("-s", "--save_file_pre",
                      action="store", dest="save_file_pre",
                      help="the prefix for saving files")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)
    main(options, args)