#!/usr/bin/env python
# Filename: select_rts_from_YOLO_demDiff_headwall.py 
"""
introduction: based on the RTS mapping results from YOLO or Deeplab, then combine with DEM diff and headwall extracted from slope,
to remove false positives.

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 01 August, 2021
"""


import os,sys
from optparse import OptionParser
machine_name = os.uname()[1]

import time
from datetime import datetime

import pandas as pd

deeplabforRS =  os.path.expanduser('~/codes/PycharmProjects/DeeplabforRS')
sys.path.insert(0, deeplabforRS)
import vector_gpd
from shapely.strtree import STRtree
import basic_src.io_function as io_function
import basic_src.basic as basic
import basic_src.timeTools as timeTools
import basic_src.map_projection as map_projection

sys.path.insert(0, os.path.expanduser('~/codes/PycharmProjects/rs_data_proc/DEMscripts'))
from dem_common import grid_20_shp, grid_dem_diffs_segment_dir, grid_dem_headwall_shp_dir, \
    grid_rts_shp_dir,grid_dem_subsidence_select

from produce_DEM_diff_ArcticDEM import get_grid_20

from multiprocessing import Pool

def get_existing_select_grid_rts(rts_shp_dir, grid_base_name, grid_ids):

    existing_grid_rts_shp = []
    grid_id_no_rts_shp = []
    for id in grid_ids:
        rts_shps_dir = io_function.get_file_list_by_pattern(rts_shp_dir, '*_grid%d'%id)
        if len(rts_shps_dir) == 1:
            existing_grid_rts_shp.append(rts_shps_dir[0])
            continue
        elif len(rts_shps_dir) > 1:
            existing_grid_rts_shp.append(rts_shps_dir[0])
            basic.outputlogMessage('warning, There are multiple rts shps for grid: %d'%id)
            for item in rts_shps_dir:
                basic.outputlogMessage(item)
            continue
        else:
            pass

        grid_id_no_rts_shp.append(id)
    if len(existing_grid_rts_shp) > 0:
        basic.outputlogMessage('%d existing grid select RTS shps for the input grid_ids or extent'%len(existing_grid_rts_shp))
    else:
        basic.outputlogMessage('no existing grid select RTS shps')
    return existing_grid_rts_shp, grid_id_no_rts_shp


def find_results_for_one_grid_id(root_dir, folder, pattern, grid_id, grid_poly, description='dem_subsidence'):
    shp_dir = os.path.join(root_dir, folder)
    shps = io_function.get_file_list_by_pattern(shp_dir, pattern)

    if len(shps) < 1:
        print('warning, no results in folder: %s, with pattern: %s'%(folder,pattern))
        return None

    if description=='dem_subsidence':
        if len(shps)==1:
            return shps[0]
        # if it's subsidence from dem diff, should only have one shp
        if len(shps) > 1:
            raise ValueError('warning, more than one result in folder: %s, with pattern: %s'%(folder,pattern))

    if description == 'headwall':
        return shps

def find_overlap_one_polygon(idx, poly, tree, count_group1):
    if idx % 1000 == 0:
        print(datetime.now(), '%d th / %d polygons' % (idx, count_group1))
    adjacent_polygons = [item for item in tree.query(poly) if
                         item.intersects(poly) or item.touches(poly)]
    if len(adjacent_polygons) > 0:
        return idx
    return None

def select_polygons_overlap_others_in_group2(polys_group1_path,polys_group2_path,save_path, buffer_size=0,process_num=1):
    '''
    select polygons in group 1 that that any polygons in group 2
    :param polys_group1_path:
    :param polys_group2_path:
    :param buffer_size:
    :param process_num: it will take a lot of memory when using multiple processs
    :return:
    '''
    if os.path.isfile(save_path):
        print('Warning, %s exists, skip'%(save_path))
        return True

    # check projections
    shp1_prj = map_projection.get_raster_or_vector_srs_info_proj4(polys_group1_path)
    if shp1_prj is False:
        raise ValueError('get projection of %s failed' % (polys_group1_path))
    shp2_prj = map_projection.get_raster_or_vector_srs_info_proj4(polys_group2_path)
    if shp2_prj is False:
        raise ValueError('get projection of %s failed' % (polys_group2_path))
    if shp1_prj != shp2_prj:
        raise ValueError('%s and %s dont have the same projection'%(polys_group1_path,polys_group2_path))


    polys_group1 = vector_gpd.read_polygons_gpd(polys_group1_path,b_fix_invalid_polygon=False)
    if buffer_size > 0:
        print(datetime.now(), 'starting buffering for %d polygons in group 1' % len(polys_group1))
        polys_group1 = [item.buffer(buffer_size) for item in polys_group1]
    print(datetime.now(),'read %d polygons in group 1'%len(polys_group1))
    polys_group2 = vector_gpd.read_polygons_gpd(polys_group2_path,b_fix_invalid_polygon=False)
    print(datetime.now(), 'read %d polygons in group 2' % len(polys_group2))

    shp1_prj = vector_gpd.get_projection(polys_group1_path,format='epsg')
    overlap_touch_gpd = vector_gpd.polygons_overlap_another_group_in_file(polys_group1, shp1_prj, polys_group2_path)
    overlap_touch_gpd.to_file(save_path)

    # count_group1 = len(polys_group1)
    # # https://shapely.readthedocs.io/en/stable/manual.html#str-packed-r-tree
    # tree = STRtree(polys_group2)
    # select_idx = []
    # if process_num == 1:
    #     for idx, subsi_buff in enumerate(polys_group1):
    #         # output progress
    #         if idx%1000 == 0:
    #             print(datetime.now(),'%d th / %d polygons'%(idx,count_group1))
    #
    #         adjacent_polygons = [item for item in tree.query(subsi_buff) if
    #                              item.intersects(subsi_buff) or item.touches(subsi_buff)]
    #         if len(adjacent_polygons) > 0:
    #             select_idx.append(idx)
    # elif process_num > 1:
    #     # end in error:
    #     # struct.error: 'i' format requires -2147483648 <= number <= 2147483647
    #     # could be caused by too many polygons
    #     raise ValueError("has error of struct.error: 'i' format requires -2147483648 <= number <= 2147483647, "
    #                      "please use process=1")
    #
    #     # theadPool = Pool(process_num)
    #     # parameters_list = [(idx, poly, tree, count_group1) for idx, poly in enumerate(polys_group1)]
    #     # results = theadPool.starmap(find_overlap_one_polygon, parameters_list)
    #     # select_idx = [item for item in results if item is not None]
    #     # theadPool.close()
    # else:
    #     raise ValueError('wrong process number: %s'%str(process_num))
    #
    #
    # basic.outputlogMessage('Select %d polygons from %d ones' % (len(select_idx), count_group1))
    #
    # if len(select_idx) < 1:
    #     return None
    #
    # # save to subset of shaepfile
    # return vector_gpd.save_shapefile_subset_as(select_idx, polys_group1_path, save_path)


def merge_polygon_for_demDiff_headwall_grids(dem_subsidence_shp, headwall_shp_list, output_dir, buffer_size=50):

    output = os.path.join(output_dir, os.path.basename(io_function.get_name_by_adding_tail(dem_subsidence_shp,'HW_select')))
    if os.path.isfile(output):
        print('warning, %s already exists'%output)
        return output

    # check projections?

    subsidence_list = vector_gpd.read_polygons_gpd(dem_subsidence_shp)
    headwall_all_list =[]
    for shp in headwall_shp_list:
        headwall_list = vector_gpd.read_polygons_gpd(shp)
        headwall_all_list.extend(headwall_list)

    # # https://shapely.readthedocs.io/en/stable/manual.html#str-packed-r-tree
    tree = STRtree(headwall_all_list)

    subsidence_buff_list = [item.buffer(buffer_size) for item in subsidence_list]
    select_idx = []
    for idx, subsi_buff in enumerate(subsidence_buff_list):

        adjacent_polygons = [item for item in tree.query(subsi_buff) if
                             item.intersects(subsi_buff) or item.touches(subsi_buff)]
        if len(adjacent_polygons) > 0:
            select_idx.append(idx)



    basic.outputlogMessage('Select %d polygons from %d ones'%(len(select_idx), len(subsidence_list)))

    if len(select_idx) < 1:
        return None

    # save to subset of shaepfile
    if vector_gpd.save_shapefile_subset_as(select_idx,dem_subsidence_shp,output) is True:
        return output


def select_rts_one_grid(mapping_res_dir, dem_subsidence_dir, grid_headwall_dir,grid_id,grid_poly):
    # find subsidence results
    dem_subsidence_shp = find_results_for_one_grid_id(dem_subsidence_dir, 'segment_result_grid%d' % grid_id,
                                                      '*_post.shp',
                                                      grid_id, grid_poly, description='dem_subsidence')
    if dem_subsidence_shp is None:
        return None

    # find headwall results
    headwall_shp_list = find_results_for_one_grid_id(grid_headwall_dir, 'headwall_shps_grid%d' % grid_id, '*.shp',
                                                     grid_id, grid_poly, description='headwall')
    if headwall_shp_list is None:
        return None

    # merge the results of these two
    output_dir = os.path.join(grid_dem_subsidence_select, 'subsidence_grid%d' % grid_id)
    if os.path.isdir(output_dir) is False:
        io_function.mkdir(output_dir)
    subsidence_poly_select_shp = merge_polygon_for_demDiff_headwall_grids(dem_subsidence_shp, headwall_shp_list,
                                                                          output_dir)
    if subsidence_poly_select_shp is None:
        return None
    # find the mapping results


    return subsidence_poly_select_shp


def select_rts_map_demDiff_headwall_grids(mapping_res_dir, dem_subsidence_dir, grid_headwall_dir, grid_polys, grid_ids, pre_name, process_num=1):
    ''' to select RTS from mapping results, dem subsidence, and headwall '''

    if process_num==1:
        # select rts polygons one by oen
        for grid_id, grid_poly in zip(grid_ids, grid_polys):
            select_poly_shp = select_rts_one_grid(mapping_res_dir, dem_subsidence_dir, grid_headwall_dir, grid_id, grid_poly)
            if select_poly_shp is None:
                continue

    elif process_num > 1:
        theadPool = Pool(process_num)
        parameters_list = [(mapping_res_dir, dem_subsidence_dir, grid_headwall_dir, grid_id, grid_poly)
                           for grid_id, grid_poly in zip(grid_ids, grid_polys)]
        results = theadPool.starmap(select_rts_one_grid, parameters_list)
    else:
        raise ValueError('wrong process_num: %s'%str(process_num))


    pass


def test_select_polygons_overlap_each_other():
    yolo_box_shp= os.path.expanduser('~/Data/Arctic/alaska/autoMapping/alaskaNS_yolov4_4/result_backup/'
                                     'alaNS_hillshadeHWline_2008to2017_alaskaNS_yolov4_4_exp4_1/alaNS_hillshadeHWline_2008to2017_alaskaNS_yolov4_4_exp4_post_1.shp')
    dem_subsidence_shp = os.path.expanduser('~/Data/dem_processing/grid_dem_diffs_segment_subsidence.gpkg')


    save_path = io_function.get_name_by_adding_tail(yolo_box_shp,'select')
    select_polygons_overlap_others_in_group2(yolo_box_shp,dem_subsidence_shp,save_path, process_num=4)


def test_merge_polygon_for_demDiff_headwall_grids():
    dem_subsidence_shp = os.path.expanduser('~/Data/tmp_data/segment_result_grid9999/ala_north_slo_extent_latlon_grid_ids_DEM_diff_grid9999_8bit_post_final.shp')
    headwall_shp_list = io_function.get_file_list_by_pattern(os.path.expanduser('~/Data/tmp_data/headwall_shps_grid9999'),
                                                         '*.shp')

    # merge the results of these two
    output_dir = os.path.join(grid_dem_subsidence_select, 'subsidence_grid%d' % 9999)
    if os.path.isdir(output_dir) is False:
        io_function.mkdir(output_dir)

    merge_polygon_for_demDiff_headwall_grids(dem_subsidence_shp, headwall_shp_list, output_dir, buffer_size = 50)

def main(options, args):

    process_num = options.process_num
    buffer_size = options.buffer_size
    # perform the selection grid by grid
    basic.setlogfile('select_RTS_YOLO_demDiff_headwall_%s.txt' % timeTools.get_now_time_str())



    b_grid = options.b_grid
    if b_grid:
        # process the selection grid by grid
        extent_shp_or_ids_txt = args[0]
        yolo_result_dir = os.path.expanduser('~/Data/Arctic/alaska/autoMapping/alaskaNS_yolov4_1')
        dem_subsidence_dir = grid_dem_diffs_segment_dir
        grid_headwall_dir = grid_dem_headwall_shp_dir

        # read grids and ids
        time0 = time.time()
        all_grid_polys, all_ids = vector_gpd.read_polygons_attributes_list(grid_20_shp, 'id')
        print('time cost of read polygons and attributes', time.time() - time0)

        # get grid ids based on input extent
        grid_base_name = os.path.splitext(os.path.basename(extent_shp_or_ids_txt))[0]
        grid_polys, grid_ids = get_grid_20(extent_shp_or_ids_txt, all_grid_polys, all_ids)

        # check dem difference existence
        grid_rts_shps, grid_id_no_rts_shp = get_existing_select_grid_rts(grid_rts_shp_dir,grid_base_name, grid_ids)

        if len(grid_id_no_rts_shp) > 0:
            # refine grid_polys
            if len(grid_ids) > len(grid_id_no_rts_shp):
                id_index = [grid_ids.index(id) for id in grid_id_no_rts_shp]
                grid_polys = [grid_polys[idx] for idx in id_index]
            #
            rts_shp_folders = select_rts_map_demDiff_headwall_grids(yolo_result_dir,dem_subsidence_dir,grid_headwall_dir,
                                                grid_polys, grid_id_no_rts_shp, grid_base_name, process_num=process_num)
    else:
        # processing the selection for two input shapefile
        yolo_box_shp = args[0]
        dem_subsidence_shp = args[1]
        print('polygon group 1:',yolo_box_shp)
        print('polygon group 2:',dem_subsidence_shp)

        if options.save_path is not None:
            save_path = options.save_path
        else:
            save_path = io_function.get_name_by_adding_tail(yolo_box_shp, 'select')

        select_polygons_overlap_others_in_group2(yolo_box_shp, dem_subsidence_shp, save_path, buffer_size=buffer_size,
                                                 process_num=process_num)

    pass



if __name__ == '__main__':
    # test_select_polygons_overlap_each_other()
    # sys.exit(0)
    usage = "usage: %prog [options] extent_shp or grid_id_list.txt OR poly_path1 poly_path2 "
    parser = OptionParser(usage=usage, version="1.0 2021-3-6")
    parser.description = 'Introduction: select mapping polygons by checking if it overlap with any polygons in another groups  '

    parser.add_option("-g", "--b_grid",
                      action="store_true", dest="b_grid", default=False,
                      help="if True, then we will process the selection grid by grids")

    parser.add_option("-b", "--buffer_size",
                      action="store", dest="buffer_size", type=float, default=0,
                      help="the buffer size for poly_path1, if group 1 are lines, buffer_size have to be set")

    parser.add_option("", "--save_path",
                      action="store", dest="save_path",
                      help="the path for saving file")

    parser.add_option("", "--process_num",
                      action="store", dest="process_num", type=int, default=4,
                      help="number of processes to create the mosaic")

    (options, args) = parser.parse_args()
    # print(options.create_mosaic)

    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    print('process grid by grid?',options.b_grid)

    main(options, args)