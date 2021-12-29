#!/usr/bin/env python
# Filename: get_valid_percent_entropy 
"""
introduction: get the valid pixel percentage and shannon_entropy for each images

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 11 February, 2021
"""

import os, sys
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import numpy as np
import matplotlib
import rasterio
import datasets.raster_io as raster_io

import basic_src.io_function as io_function
import basic_src.basic as basic


# must be before importing matplotlib.pyplot or pylab!
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets.data_figures import histogram2logfile

def np_histogram_a_list(in_list,bin_count=255,axis_range=(0, 10)):
    np_array = np.array(in_list)
    hist, bin_edges = np.histogram(np_array, bins=bin_count, density=False, range=axis_range)

    return hist, bin_edges

def plot_valid_entropy(in_folder, save_file_pre=None):

    if save_file_pre is None:
        save_file_pre  = os.path.basename(in_folder)

    basic.setlogfile(save_file_pre + 'hist_info.txt')
    image_paths = io_function.get_file_list_by_ext('.tif', in_folder, bsub_folder=True)
    if len(image_paths) < 1:
        raise IOError('no tif files in %s' % in_folder)
    valid_per_list = []
    entropy_list = []
    img_count = len(image_paths)
    for idx, img_path in enumerate(image_paths):
        print('%d/%d'%(idx+1, img_count))
        valid_per, entropy = raster_io.get_valid_percent_shannon_entropy(img_path, log_base=10)
        valid_per_list.append(valid_per)
        entropy_list.append(entropy)

    per_entropy_txt = save_file_pre + '_' + 'valid_per_entropy.txt'
    save_hist_path = save_file_pre +'_' + 'hist.jpg'
    with open(per_entropy_txt, 'w') as f_obj:
        for path, per, entropy in zip(image_paths, valid_per_list,entropy_list):
            f_obj.writelines(os.path.basename(path) + ' %.4f  %.6f \n'%(per, entropy))

    # plot the histogram
    fig = plt.figure(figsize=(6,4)) #
    ax1 = fig.add_subplot(111)
    n, bins, patches = plt.hist(x=entropy_list, bins=50, color='b', rwidth=0.85)
    # print(n, bins, patches)
    plt.savefig(save_hist_path, dpi=200)  # 300
    histogram2logfile(entropy_list,bins,hist_tag=save_hist_path)

    return save_hist_path

def main(options, args):
    in_folder = args[0]

    save_file_pre = options.save_file_pre
    plot_valid_entropy(in_folder, save_file_pre)



if __name__ == '__main__':

    usage = "usage: %prog [options] image_folder  "
    parser = OptionParser(usage=usage, version="1.0 2021-02-11")
    parser.description = 'Introduction:  get the valid pixel percentage and shannon_entropy for each image'


    parser.add_option("-s", "--save_file_pre",
                      action="store", dest="save_file_pre",
                      help="the prefix for saving files")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)
    main(options, args)