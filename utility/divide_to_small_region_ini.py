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
import basic_src.map_projection as map_projection
import basic_src.basic as basic
import datasets.raster_io as raster_io
import datasets.vector_gpd as vector_gpd

grid_20_id_raster = os.path.expanduser('~/Data/Arctic/ArcticDEM/grid_shp/ArcticDEM_grid_20km_id.tif')

import re
import numpy as np
import pandas as pd

def modify_parameter(para_file, para_name, new_value):
    parameters.write_Parameters_file(para_file,para_name,new_value)

# copy from dem_common.py
def get_grid_id_from_path(item):
    return int(re.findall(r'grid\d+', os.path.basename(item))[0][4:])


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

def find_connect_grids(grid_polys, grid_ids, max_grid_count, grid_ids_2d, visit_np, save_path, proj=None):
    # find a connected region with for processing, and save to files
    seed_loc = np.where(visit_np == 0)
    if len(seed_loc[0]) < 1:
        print('warning, all pixels have been visited')
        return [], []
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

    select_grid_polys = [grid_polys[grid_ids.index(item)] for item in selected_gird_id_list]

    return select_grid_polys, selected_gird_id_list

def find_sub_regions(grid_polys, grid_ids, min_grid_count, max_grid_count, grid_ids_2d, visit_np, save_path, proj=None):

    select_grid_polys, selected_gird_id_list = [], []
    while len(select_grid_polys) < min_grid_count:
        grid_polys_con, gird_id_list_con = find_connect_grids(grid_polys, grid_ids, max_grid_count, grid_ids_2d, visit_np, save_path, proj=None)
        if len(gird_id_list_con) < 1:
            break
        select_grid_polys.extend(grid_polys_con)
        selected_gird_id_list.extend(gird_id_list_con)
        if len(selected_gird_id_list) >= max_grid_count:
            break

    if len(selected_gird_id_list) > 0:
        save_selected_girds_and_ids(selected_gird_id_list, select_grid_polys, proj, save_path)
    return select_grid_polys, selected_gird_id_list

def divide_large_region_into_subsets(in_grid_shp, save_dir, min_grid_count=20, max_grid_count=200):

    gird_prj = map_projection.get_raster_or_vector_srs_info_proj4(in_grid_shp)
    # read grids
    grid_polys, grid_ids = vector_gpd.read_polygons_attributes_list(in_grid_shp,'cell_id')
    # print(grid_ids)
    # burn into a np array
    grid_ids_2d = raster_io.burn_polygons_to_a_raster(grid_20_id_raster, grid_polys, grid_ids, None, date_type='int32')
    # print('grid_ids_2d', grid_ids_2d.shape, np.min(grid_ids_2d), np.max(grid_ids_2d))

    visit_np = np.zeros_like(grid_ids_2d, dtype=np.uint8)
    visit_np[ grid_ids_2d == 0] = 1          # marked those data data region as visited

    subset_id = 0
    while True:
        select_grids_shp = os.path.join(save_dir, io_function.get_name_no_ext(in_grid_shp) + '_sub%d' % subset_id + '.shp')
        # when re-run this, each subset will be the same or some grids in the subset would be removed if they has been completed (or ignored)
        select_grid_polys, selected_gird_ids = find_sub_regions(grid_polys, grid_ids, min_grid_count,
                                                                              max_grid_count,
                                                                              grid_ids_2d, visit_np, select_grids_shp,
                                                                              proj=gird_prj)

        # print('subset_id: %d, find %d grids'%(subset_id, len(selected_gird_ids)))
        subset_id += 1

        if len(select_grid_polys) < 1:
            break


def test_divide_large_region_into_subsets():
    grids_shp = os.path.expanduser('~/Data/slump_demdiff_classify/select_regions_Huangetal2023/overlap_touch_all.shp')
    save_dir = 'sub-regions'
    if os.path.isdir(save_dir) is False:
        io_function.mkdir(save_dir)
    divide_large_region_into_subsets(grids_shp,save_dir)


def create_a_region_defined_parafile(template_para_file, grid_ids_txt, img_list, img_grid_id_list, save_dir=None):

    io_function.is_file_exist(template_para_file)
    if save_dir is None:
        save_dir = os.path.dirname(grid_ids_txt)
    if os.path.isdir(save_dir) is False:
        io_function.mkdir(save_dir)

    sub_id_str = re.findall(r'sub\d+', os.path.basename(grid_ids_txt))[0]

    new_para_file = os.path.join(save_dir, os.path.basename(io_function.get_name_by_adding_tail(template_para_file, sub_id_str)))

    # copy the file
    io_function.copy_file_to_dst(template_para_file,new_para_file)

    # add sub_id_str to area_remark
    area_remark = parameters.get_string_parameters(template_para_file, 'area_remark')
    modify_parameter(new_para_file, 'area_remark', area_remark+'_'+sub_id_str)

    grid_ids = [int(item.strip())  for item in io_function.read_list_from_txt(grid_ids_txt)]

    select_img_list = [ img_p for img_p, grid_id in zip(img_list,img_grid_id_list) if grid_id in grid_ids]

    # create links in the sub folder
    img_save_dir = os.path.join(save_dir,sub_id_str+'_images')
    io_function.mkdir(img_save_dir)
    for img_path in select_img_list:
        target = os.path.join(img_save_dir, os.path.basename(img_path))
        cmd_str = 'ln -s %s %s'%(img_path, target)
        basic.os_system_exit_code(cmd_str)

    # modify inf_image_dir and inf_image_or_pattern
    modify_parameter(new_para_file, 'inf_image_dir', img_save_dir)

    return new_para_file


def divide_large_region_ini_into_subsets_ini(region_ini, region_grid_shp, min_grid_count, max_grid_count, save_dir = None):
    '''
    divide a large region into many subsets, for parallel running image classification
    :param region_ini: the region ini
    :param region_grid_shp: the grid shapefile (e.g. 20 by 20 km grids), corresponding to this region_ini
    :param min_grid_count: min grid count for each subset
    :param max_grid_count: max grid count for each subset
    :param save_dir: output dir
    :return:
    '''

    area_name_remark_time = parameters.get_area_name_remark_time(region_ini)
    if save_dir is None:
        save_dir = area_name_remark_time + '_sub_regions'
    if os.path.isdir(save_dir):
        io_function.mkdir(save_dir)

    # the images (per grid) for this region
    inf_image_dir = parameters.get_directory(region_ini, 'inf_image_dir')
    # it is ok consider a file name as pattern and pass it the following functions to get file list
    inf_image_or_pattern = parameters.get_string_parameters(region_ini, 'inf_image_or_pattern')
    inf_img_list = io_function.get_file_list_by_pattern(inf_image_dir, inf_image_or_pattern)
    img_count = len(inf_img_list)
    if img_count < 1:
        raise ValueError(
            'No image for inference, please check inf_image_dir (%s) and inf_image_or_pattern (%s) in %s'
            % (inf_image_dir, inf_image_or_pattern, region_ini))

    inf_img_grid_id_list = [get_grid_id_from_path(img_path) for img_path in  inf_img_list ]
    grid_polys = vector_gpd.read_polygons_gpd(region_grid_shp,b_fix_invalid_polygon=False)

    if len(inf_img_grid_id_list) != len(grid_polys):
        raise ValueError('the image count (%d) and grid count (%d) is different'%(len(inf_img_grid_id_list), len(grid_polys)))

    # divide to sub-regions
    divide_large_region_into_subsets(region_grid_shp, save_dir,  min_grid_count=min_grid_count, max_grid_count=max_grid_count)
    grid_ids_txt_list = io_function.get_file_list_by_pattern(save_dir,'*grid_ids.txt')

    # create area ini files for each sub-regions
    sub_region_ini_files_list = []

    for grid_id_txt in grid_ids_txt_list:
        new_area_ini = create_a_region_defined_parafile(region_ini,grid_id_txt, inf_img_list, inf_img_grid_id_list, save_dir)
        sub_region_ini_files_list.append(new_area_ini)

    with open('%s_region_ini_files.txt'%os.path.basename(save_dir),'w') as f_obj:
        for ini in sub_region_ini_files_list:
            f_obj.writelines(os.path.abspath(ini) + '\n')




def main(options, args):
    big_region_ini = args[0]
    region_grid_shp = args[1]
    max_grid_count_per_ini = options.max_grid_count
    min_grid_count_per_ini = options.min_gird_count
    save_dir = options.save_dir

    divide_large_region_ini_into_subsets_ini(big_region_ini, region_grid_shp, min_grid_count_per_ini, max_grid_count_per_ini,  save_dir)


if __name__ == '__main__':

    # test_divide_large_region_into_subsets()
    # sys.exit(0)

    usage = "usage: %prog [options]  big_region_ini grids_shp"
    parser = OptionParser(usage=usage, version="1.0 2024-04-23")
    parser.description = 'Introduction: divide a big region into many small regions (ini) '

    parser.add_option("-l", "--min_gird_count",
                      action="store", dest="min_gird_count", type=int, default=20,
                      help="minimum image (grid) count per sub regions")

    parser.add_option("-u", "--max_grid_count",
                      action="store", dest="max_grid_count", type=int, default=200,
                      help="maximum image (grid) count per sub regions")

    parser.add_option("-s", "--save_dir",
                      action="store", dest="save_dir",
                      help="the folder to save the ini files of sub regions")


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 2:
        parser.print_help()
        sys.exit(2)
    main(options, args)