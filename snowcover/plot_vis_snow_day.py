#!/usr/bin/env python
# Filename: plot_snow_timeseries.py
"""
introduction: visualization of 2D snow cover days

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 22 May, 2019
"""

import sys,os
from optparse import OptionParser
import rasterio
import numpy as np

# import pandas as pd # read and write excel files

HOME = os.path.expanduser('~')
codes_dir2 = HOME + '/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)

import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import csv


def plot_one_figure(ax,fig_path,colormap=None,min_val=0, max_val=30, top_label = None):

    img = mpimg.imread(fig_path)

    # default disalbe all tick
    ax.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=False,
        labeltop=False, labelleft=False, labelright=False)  # labels are off

    fontsize = 12
    # drop lable
    # draw bottom tick and label
    if top_label is not None:
        # ax.tick_params(axis='x', bottom=True, labelbottom=True)
        # x_tick_pos = np.arange(0, width, float(width) / 6)
        # x_tick_pos = np.delete(x_tick_pos,0)    # remove the first one
        # ax.set_xticks(x_tick_pos)
        # #
        # x_labels = np.arange(1, 6, 1)
        # ax.set_xticklabels(x_labels)

        ax.set_xlabel(top_label, fontsize=fontsize)
        ax.xaxis.set_label_position('top')

    im = ax.imshow(img, cmap=colormap, vmin=min_val, vmax=max_val)  # jet_r
    # ax.imshow(img)  # jet_r

    return im

def get_year_month(file_name):

    fn = os.path.splitext(os.path.basename(file_name))[0]
    tmp_list = fn.split('_')

    return '-'.join(tmp_list[2:])

def draw_multi_snow_days(snow_days_file_list, output):

    # print(snow_days_file_list)

    ncount = len(snow_days_file_list)
    if ncount != 10:
        raise ValueError('Currently only accept 10 figures')

    # 12 inch by 8 inch, adjust the figure size to change the space between sub figures
    fig = plt.figure(figsize=(17, 3.5))

    # shape = (2,4) # 2 row, 4 column
    # ax = plt.subplot2grid((1,1),(0,0))
    # plot_one_figure(ax,r0_list[0],l_tick=True,b_tick=True)

    grid = gridspec.GridSpec(2, 5, figure=fig)
    # print(grid)
    grid.update(wspace=0.02, hspace=0.02)  # set the spacing between axes.

    top_labels = [get_year_month(file_name) for file_name in  snow_days_file_list ]

    colormap = 'jet'
    my_cmap = cm.get_cmap(colormap)  # 'jet_r' bwr_r
    #
    # # for yearly
    min_val = 0
    max_val = 70
    norm = colors.Normalize(min_val, max_val)
    # my_cmap.set_over((1, 1, 1))  # set the color greater than max_val
    # my_cmap.set_under((0, 0, 0))  # set the color less than min_val

    im_shown = None

    for idx, file_name in enumerate(snow_days_file_list):
        # ax = plt.subplot2grid(shape, (0, idx))
        # print(grid[idx])
        ax = plt.subplot(grid[idx])
        # example_plot(ax)

        # plot_one_figure(ax, file_name, colormap=my_cmap, min_val=0, max_val=30, top_label=top_labels[idx])

        im_shown = plot_one_figure(ax, file_name, colormap=my_cmap, min_val=min_val, max_val=max_val, top_label=top_labels[idx])

        # break

    fig.subplots_adjust(right=0.8)

    #The dimensions [left, bottom, width, height] of the new axes.
    # All quantities are in fractions of figure width and height.
    cbar_ax = fig.add_axes([0.81, 0.15, 0.01, 0.7])
    fig.colorbar(im_shown, cax=cbar_ax)


    # plt.tight_layout()

    # show figure is correct, but the space is wrong in the saved one
    # plt.show()
    plt.savefig(output, bbox_inches="tight", dpi=200)
    print('save to %s' % output)


def main(options, args):

    snow_days_file_list = args


    output = options.output

    draw_multi_snow_days(snow_days_file_list, output)





if __name__ == "__main__":
    usage = "usage: %prog [options] snow_day_file1 snow_day_file2 ..."
    parser = OptionParser(usage=usage, version="1.0 2019-5-22")
    parser.description = 'Introduction: plot multiple maps of snow days together'

    parser.add_option("-o", "--output",
                      action="store", dest="output",
                      help="the output file path")


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    ## set parameters files
    # if options.para_file is None:
    #     print('error, no parameters file')
    #     parser.print_help()
    #     sys.exit(2)
    # else:
    #     parameters.set_saved_parafile_path(options.para_file)

    main(options, args)
