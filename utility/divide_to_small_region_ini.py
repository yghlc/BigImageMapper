#!/usr/bin/env python
# Filename: create_region_ini.py
"""
introduction:  divide a big region in (area*.ini) into many small regions for parallel prediction (image classification)

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 23 April, 2024
"""

import os, sys
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters

import basic_src.io_function as io_function
import basic_src.basic as basic
import datasets.raster_io as raster_io
import datasets.vector_gpd as vector_gpd

grid_20_id_raster = os.path.expanduser('~/Data/Arctic/ArcticDEM/grid_shp/ArcticDEM_grid_20km_id.tif')

import re
import numpy as np
import pandas as pd

def modify_parameter(para_file, para_name, new_value):
    parameters.write_Parameters_file(para_file,para_name,new_value)


def find_neighbours_2d(grid_ids_2d,visited,seed,connect):
    '''
    find neighbourhood voxels
    :param grid_ids_2d: 2D data
    :param visited: indicates pixels has been checked
    :param seed: a seed
    :param connect: pixel connectivity
    :return: a list containing voxels
    '''

    height, width = grid_ids_2d.shape
    y,x = seed[0],seed[1]
    visited[y, x] = 1

    neigh_range = [-1,0,1]
    neighbours = [[i,j] for i in neigh_range for j in neigh_range  ]
    neighbours.remove([0,0])

    # distance within 1
    if connect==4:
        connectivity =  [ [y+dy, x+dx] for (dy,dx) in neighbours if (dy*dy + dx*dx) <= 1 ]
    # distance within sqrt(2)
    elif connect==8:
        connectivity = [[y + dy, x + dx] for (dy, dx) in neighbours if (dy * dy + dx * dx) <= 2]
    else:
        raise ValueError('Only accept connectivity of 4 or 8')

    new_seeds = []
    for [y,x] in connectivity:
        # out extent
        if y<0 or x<0 or y >= height or x >= width:
            continue
        # already visited
        if visited[y,x]:
            continue
        new_seeds.append([y,x])

        # masked as visited
        visited[y,x] = 1

    return new_seeds

def save_selected_girds_and_ids(selected_gird_id_list,select_grid_polys,proj,save_path):
    # save to shapefile to download and processing
    # change numpy.uint16 to int, avoid become negative when saving to shapefile
    selected_gird_id_list = [int(item) for item in selected_gird_id_list]
    save_pd = pd.DataFrame({'grid_id':selected_gird_id_list, 'Polygon':select_grid_polys})
    vector_gpd.save_polygons_to_files(save_pd,'Polygon',proj,save_path)
    basic.outputlogMessage('saved %d grids to %s'%(len(select_grid_polys), save_path))
    # save the ids to txt
    save_id_txt = os.path.splitext(save_path)[0] + '_grid_ids.txt'
    selected_grid_ids_str = [str(item) for item in selected_gird_id_list]
    io_function.save_list_to_txt(save_id_txt, selected_grid_ids_str)

def find_connect_grids(grid_polys, grid_ids, ignore_ids,max_grid_count, grid_ids_2d, visit_np, save_path, proj=None):
    # find a connected region with for donwload and processing, and save to files
    seed_loc = np.where(visit_np == 0)
    if len(seed_loc[0]) < 1:
        print('warning, all pixels have been visited')
        return None, None
    # seed_loc = np.where(grid_ids_2d == grid_ids[0])
    y, x = seed_loc[0][0], seed_loc[1][0]
    selected_gird_id_list = [grid_ids_2d[y, x]]
    seed_list = [[y, x]]
    while len(selected_gird_id_list) < max_grid_count and len(seed_list) > 0:
        # find neighbours
        new_seeds = find_neighbours_2d(grid_ids_2d, visit_np, seed_list[0], 8)
        del seed_list[0]
        seed_list.extend(new_seeds)
        # find new ids
        for seed in new_seeds:
            row, col = seed
            selected_gird_id_list.append(grid_ids_2d[row, col])

    # remove some ids
    # selected_gird_id_list = [id for id in selected_gird_id_list if id not in ignore_ids]
    ignore_ids_in_selected = list(set(ignore_ids).intersection(selected_gird_id_list))  # intersection is faster
    _ = [selected_gird_id_list.remove(rm_id) for rm_id in ignore_ids_in_selected]
    if len(selected_gird_id_list) < 1:
        return [], []

    select_grid_polys = [grid_polys[grid_ids.index(item)] for item in selected_gird_id_list]

    save_selected_girds_and_ids(selected_gird_id_list, select_grid_polys, proj, save_path)

    return select_grid_polys, selected_gird_id_list

def divide_large_region_into_subsets(in_grid_shp, save_dir):

    # read grids
    grid_polys, grid_ids = vector_gpd.read_polygons_attributes_list(in_grid_shp,'cell_id')
    # burn into a np array
    grid_ids_2d = raster_io.burn_polygons_to_a_raster(grid_20_id_raster, grid_polys, 1, None)

    visit_np = np.zeros_like(grid_ids_2d, dtype=np.uint8)

    subset_id = -1
    while True:
        subset_id += 1
        select_grids_shp = os.path.join(save_dir,
                                        io_function.get_name_no_ext(in_grid_shp) + '_sub%d' % subset_id + '.shp')
        # when re-run this, each subset will be the same or some grids in the subset would be removed if they has been completed (or ignored)
        select_grid_polys, selected_gird_ids = get_grids_for_download_process(grid_polys, grid_ids, ignore_ids,
                                                                              max_grid_count,
                                                                              grid_ids_2d, visit_np, select_grids_shp,
                                                                              proj=gird_prj)



def test_divide():
    grids_shp = os.path.expanduser('~/Data/slump_demdiff_classify/select_regions_Huangetal2023/overlap_touch_all.shp')
    save_dir = './'
    divide_large_region_into_subsets(grids_shp)



def main(options, args):
    big_region_ini = args[0]
    max_grid_count_per_ini = options.max_img_count
    save_dir = options.save_dir



    image_paths = io_function.get_file_list_by_ext(ext_name,in_folder, bsub_folder=True)
    if len(image_paths) < 1:
        raise IOError('no tif files in %s'%in_folder)

    b_per_image_per_ini = options.b_per_image_per_ini
    region_ini_files_list = []
    if b_per_image_per_ini is False:
        # get unique dir list
        img_dir_list = [ os.path.dirname(item) for item in image_paths ]
        img_dir_list = set(img_dir_list)

        for img_dir in img_dir_list:
            # copy template file
            out_ini = create_new_region_defined_parafile(template_ini,img_dir,ext_name,area_name=options.area_name,
                                                           area_time=options.area_time, area_remark=options.area_remark)
            region_ini_files_list.append(out_ini)
    else:
        for img_path in image_paths:
            out_ini = create_region_parafile_for_one_image(template_ini,img_path,area_name=options.area_name,
                                                           area_time=options.area_time, area_remark=options.area_remark)
            region_ini_files_list.append(out_ini)


    with open('region_ini_files.txt','a') as f_obj:
        for ini in region_ini_files_list:
            f_obj.writelines(os.path.abspath(ini) + '\n')

    pass

if __name__ == '__main__':

    usage = "usage: %prog [options]  big_region_ini "
    parser = OptionParser(usage=usage, version="1.0 2024-04-23")
    parser.description = 'Introduction: divide a big region into many small regions (ini) '

    parser.add_option("-c", "--max_img_count",
                      action="store", dest="max_img_count", type=int,
                      help="maximum image (grid) count per sub regions")

    parser.add_option("-s", "--save_dir",
                      action="store", dest="save_dir",
                      help="the folder to save the ini files of sub regions")


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 2:
        parser.print_help()
        sys.exit(2)
    main(options, args)