#!/usr/bin/env python
# Filename: train_test_pipeline_rareplanes.py 
"""
introduction: modified from https://github.com/yghlc/yoltv4/blob/master/notebooks/train_test_pipeline.ipynb

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 01 April, 2021
"""
import os,sys

from shapely.affinity import translate
from shapely.geometry import box
import matplotlib.pyplot as plt
from shapely.wkt import loads
from importlib import reload
import geopandas as gpd
from tqdm import tqdm
import pandas as pd
import numpy as np
import collections
import skimage.io
import shapely
import random
import shutil
# import imp
import cv2
import sys
import os

yoltv4_path =  os.path.expanduser('~/codes/PycharmProjects/yghlc_yoltv4') # '/local_data/cosmiq/src/yoltv4/'
sys.path.append(os.path.join(yoltv4_path, 'yoltv4'))
import prep_train
import tile_ims_labels
import post_process
reload(prep_train)

# set dataset path
data_root = os.path.expanduser('~/Data/objection_detection/RarePlanes')  # '/local_data/cosmiq/wdata/rareplanes'

# We want the labels in the form of a geodataframe, which can be loaded via:
# gdf_pix = gpd.read_file(path_to_labels)
gdf_pix = gpd.read_file(os.path.join(data_root,'train/geojson_aircraft_tiled/59_104001001DC7F200_tile_1762.geojson'))

# Extract the cutouts from the image, centered on the aircraft, modulo some jitter
def extract_cutouts_from_images():
    # get cutouts for all images in the training corpus

    df_polys = gdf_pix.copy()
    yolt_image_ext = '.jpg'
    pop = 'train'

    outdir_root = os.path.join(data_root, pop, 'yoltv4')
    im_dir = os.path.join(data_root, pop, 'PS-RGB_tiled')  # images
    im_list = [z for z in os.listdir(im_dir) if z.endswith('.png')]  # '.tif'
    verbose = True
    super_verbose = False

    # outputs
    outdir_ims = os.path.join(outdir_root, 'images')
    outdir_labels = os.path.join(outdir_root, 'labels')
    outdir_yolt_plots = os.path.join(outdir_root, 'yolt_plot_bboxes')
    print("outdir_ims:", outdir_ims)
    # make dirs
    for z in (outdir_ims, outdir_labels, outdir_yolt_plots):
        os.makedirs(z)

    # extract cutouts and labels
    for i, im_name in enumerate(im_list):
        im_path = os.path.join(im_dir, im_name)
        print(i, "/", len(im_list), im_name)
        prep_train.yolt_from_df(im_path, df_polys,
                                window_size=416,
                                jitter_frac=0.2,
                                min_obj_frac=0.6,
                                max_obj_count=100000,
                                geometry_col='geometry',
                                category_col='category',
                                image_fname_col='image_name',
                                outdir_ims=outdir_ims,
                                outdir_labels=outdir_labels,
                                outdir_yolt_plots=outdir_yolt_plots,
                                max_plots=5,
                                yolt_image_ext=yolt_image_ext,
                                verbose=verbose, super_verbose=super_verbose)


# visualize some of the outputs cutouts
def visualize_cutouts():
    dir_tmp = outdir_ims
    rows, cols = 2, 3
    figsize = 8
    paths_tmp = [os.path.join(dir_tmp, j) for j in os.listdir(dir_tmp) if j.endswith('.jpg')]
    rand_selection = random.sample(paths_tmp, rows * cols)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(figsize * cols, figsize * rows))
    for i in range(rows * cols):
        ax = axes.flatten()[i]
        im_path_tmp = rand_selection[i]
        im_name_tmp = os.path.basename(im_path_tmp)
        im = skimage.io.imread(im_path_tmp)
        ax.imshow(im)
        ax.set_title(im_name_tmp)

# visualize some of the yolt labels
def visualize_labels():
    dir_tmp = outdir_yolt_plots
    rows, cols = 2, 3
    figsize = 8
    paths_tmp = [os.path.join(dir_tmp, j) for j in os.listdir(dir_tmp) if j.endswith('.jpg')]
    rand_selection = random.sample(paths_tmp, rows * cols)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(figsize * cols, figsize * rows))
    for i in range(rows * cols):
        ax = axes.flatten()[i]
        im_path_tmp = rand_selection[i]
        im_name_tmp = os.path.basename(im_path_tmp)
        im = skimage.io.imread(im_path_tmp)
        ax.imshow(im)
        ax.set_title(im_name_tmp)


# 1B. Create text files for training
def create_text_file_for_training():
    # set  paths
    pop = 'train'
    n_classes = 30
    outdir_root = os.path.join(data_root, pop, 'yoltv4')
    txt_dir = os.path.join(outdir_root, 'txt')
    os.makedirs(txt_dir)

    names_path = os.path.join(txt_dir, 'rareplanes.name')
    train_list_path = os.path.join(txt_dir, 'rareplanes_train_images_list.txt')
    valid_list_path = os.path.join(txt_dir, 'rareplanes_valid_images_list.txt')
    dot_data_path = os.path.join(txt_dir, 'rareplanes_train.data')

    print("names_path:", names_path)
    print("train_list_path:", train_list_path)
    print("valid_list_path:", valid_list_path)
    print("dot_data_path:", dot_data_path)



    valid_iter = 5  # means every 5th item is in valid set

    # yolt outputs
    outdir_ims = os.path.join(outdir_root, 'images')
    outdir_labels = os.path.join(outdir_root, 'labels')

    im_list_tot = sorted([os.path.join(outdir_ims, z) for z in os.listdir(outdir_ims) if z.endswith('.jpg')])
    # make train and valid_list
    im_list_train, im_list_valid = [], []
    for i, im_path in enumerate(im_list_tot):
        if (i % valid_iter) == 0:
            im_list_valid.append(im_path)
        else:
            im_list_train.append(im_path)
    # print("len im_list_train:", len(im_list_train))
    # print("len im_list_valid:", len(im_list_valid))

    # create txt files of image paths
    for list_tmp, outpath_tmp in [[im_list_train, train_list_path], [im_list_valid, valid_list_path]]:
        df_tmp = pd.DataFrame({'image': list_tmp})
        df_tmp.to_csv(outpath_tmp, header=False, index=False)

    # print("outpath_tmp:", outpath_tmp)
    # !head {outpath_tmp}


# !echo 'classes = ' {n_classes} > {dot_data_path}
# !echo 'train = ' {train_list_path} >> {dot_data_path}
# !echo 'valid = ' {valid_list_path} >> {dot_data_path}
# !echo 'names = ' {names_path} >> {dot_data_path}
# !echo 'backup = backup/' >> {dot_data_path}
# !cat {dot_data_path}


# 1C. Copy labels to training image directory
def copy_labels_to_training_imagery_dir():
    pop = 'train'
    outdir_root = os.path.join(data_root, pop, 'yoltv4')
    outdir_ims = os.path.join(outdir_root, 'images')
    outdir_labels = os.path.join(outdir_root, 'labels')

    # copy
    for f in os.listdir(outdir_labels):
        if f.endswith('.txt'):
            shutil.copy(os.path.join(outdir_labels, f), outdir_ims)


# 1D. Set up the .cfg file



def main():
    extract_cutouts_from_images()
    pass

if __name__ == '__main__':
    main()
    pass
