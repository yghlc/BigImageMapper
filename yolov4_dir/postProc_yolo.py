#!/usr/bin/env python
# Filename: postProc_yolo.py
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 08 April, 2021
"""
import os, sys
import time
from datetime import datetime
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters
import basic_src.map_projection as map_projection
import basic_src.io_function as io_function

sys.path.insert(0, os.path.join(code_dir, 'datasets'))
from merge_shapefiles import merge_shape_files

import vector_gpd
import raster_io

from datasets.get_polygon_attributes import add_polygon_attributes
from datasets.remove_mappedPolygons import remove_polygons_main
from datasets.evaluation_result import evaluation_polygons
import utility.eva_report_to_tables as eva_report_to_tables

from workflow.postProcess import group_same_area_time_observations
from workflow.postProcess import get_observation_save_dir_shp_pre
from workflow.postProcess import get_occurence_for_multi_observation

from yoltv4Based.yolt_func import convert_reverse
from yoltv4Based.yolt_func import non_max_suppression


import rasterio
import pandas as pd
import numpy as np

def pixel_xy_to_geo_xy(x0,y0, transform):
    # pixel to geo XY
    # https://rasterio.readthedocs.io/en/latest/topics/georeferencing.html
    x0_geo = transform[0] * x0 + transform[1] * y0 + transform[2]
    y0_geo = transform[3] * x0 + transform[4] * y0 + transform[5]
    return x0_geo, y0_geo

def one_box_yoloXY2imageXY(img_path, object_list,  boundary=None):
    with rasterio.open(img_path) as src:
        if boundary is None:
            transform = src.transform
            height, width = src.height, src.width
        else:
            window = raster_io.boundary_to_window(boundary)
            transform = src.window_transform(window)
            height, width = boundary[2], boundary[3]    # xsize, ysize

        class_id_list = []
        name_list = []
        confidence_list = []
        box_poly_list = []

        for object in object_list:
            # yolo xy to pixel coordinates
            box_coord = object['relative_coordinates']
            box = [box_coord['center_x'], box_coord['center_y'], box_coord['width'], box_coord['height']]

            class_id_list.append(object['class_id'])
            name_list.append(object['name'])
            confidence_list.append(object['confidence'])

            # x0, x1, y0, y1
            x0, x1, y0, y1 = convert_reverse((width,height),box)

            x0_geo, y0_geo = pixel_xy_to_geo_xy(x0,y0,transform)
            x1_geo, y1_geo = pixel_xy_to_geo_xy(x1,y1,transform)

            #  minX, minY, maxX, maxY that is: bounds:
            # remove, will convert to polygons after non_max_suppression
            # box_poly = vector_gpd.convert_image_bound_to_shapely_polygon([x0_geo,y0_geo,x1_geo,y1_geo])
            # box_poly_list.append([box_poly])

            # because Y direction in geo is opposite to the in pixel, so y0_geo > y1_geo

            box_poly_list.append([x0_geo,y1_geo,x1_geo,y0_geo])

        return class_id_list, name_list, confidence_list, box_poly_list, img_path



def boxes_yoloXY_to_imageXY(idx, total, yolo_dict, ref_image=None):
    # return: class_id_list, name_list, confidence,box_poly_list
    objects = yolo_dict['objects']
    if len(objects) < 1:
        return [], [], [], []
    if ref_image is None:
        ref_image = yolo_dict['filename']

    return one_box_yoloXY2imageXY(ref_image, objects, boundary=None)

def boxes_minXYmaxXY_to_imageXY(idx, total, json_file, ref_image_src):
    # return: class_id_list, name_list, confidence,box_poly_list
    # ref_image_src is open rasterio image object.
    objects = io_function.read_dict_from_txt_json(json_file)
    if objects is None or len(objects) < 1:
        return [],[],[],[]

    class_id_list = []
    name_list = []
    confidence_list = []
    box_poly_list = []
    transform = ref_image_src.transform

    for object in objects:

        [xmin, ymin, xmax, ymax] = object['bbox']

        class_id_list.append(object['class_id'])
        name_list.append(object['name'])
        confidence_list.append(object['confidence'])

        x0_geo, y0_geo = pixel_xy_to_geo_xy(xmin, ymin, transform)
        x1_geo, y1_geo = pixel_xy_to_geo_xy(xmax, ymax, transform)

        #  minX, minY, maxX, maxY that is: bounds
        # because Y direction in geo is opposite to the in pixel, so y0_geo > y1_geo
        box_poly_list.append([x0_geo, y1_geo, x1_geo, y0_geo])

    return class_id_list, name_list, confidence_list, box_poly_list

def yolo_results_to_shapefile(curr_dir,img_idx, area_save_dir, test_id):

    img_save_dir = os.path.join(area_save_dir, 'I%d' % img_idx)
    res_yolo_json = img_save_dir + '_result.json'
    res_json_files = []
    if os.path.isfile(res_yolo_json):
        print('found %s in %s, will get shapefile from it'%(res_yolo_json, area_save_dir))
    else:
        if os.path.isdir(img_save_dir):
            res_json_files = io_function.get_file_list_by_ext('.json',img_save_dir,bsub_folder=False)
            if len(res_json_files) < 1:
                print('Warning, no YOLO results in %s, skip'%(img_save_dir))
                return None

            print('found %d json files for patches in %s, will get shapefile from them' % (len(res_json_files),img_save_dir))
        else:
            print('Warning, folder: %s doest not exist, skip'%img_save_dir)
            return None


    out_name = os.path.basename(area_save_dir) + '_' + test_id

    # to shapefile
    out_shp = 'I%d'%img_idx + '_' + out_name + '.shp'
    out_shp_path = os.path.join(img_save_dir, out_shp)
    if os.path.isfile(out_shp_path):
        print('%s already exist' % out_shp_path)
    else:
        class_id_list = []
        name_list = []
        box_bounds_list = []
        confidence_list = []
        source_image_list = []

        if len(res_json_files) < 1:
            # use the result in *_result.json
            yolo_res_dict_list = io_function.read_dict_from_txt_json(res_yolo_json)
            total_frame = len(yolo_res_dict_list)
            image1 = yolo_res_dict_list[0]['filename']
            for idx, res_dict in enumerate(yolo_res_dict_list):
                id_list, na_list, con_list, box_list, image1 = boxes_yoloXY_to_imageXY(idx, total_frame, res_dict, ref_image=None)
                class_id_list.extend(id_list)
                name_list.extend(na_list)
                confidence_list.extend(con_list)
                box_bounds_list.extend(box_list)
                source_image_list.extend( [os.path.basename(image1)]*len(box_list) )
        else:
            # use the results in I0/*.json
            image1 = io_function.read_list_from_txt(os.path.join(area_save_dir, '%d.txt'%img_idx))[0]
            total_frame = len(res_json_files)   # the patch numbers

            # only open image once
            with rasterio.open(image1) as src:
                for idx, f_json in enumerate(res_json_files):
                    id_list, na_list, con_list, box_list = boxes_minXYmaxXY_to_imageXY(idx, total_frame,f_json,src)
                    class_id_list.extend(id_list)
                    name_list.extend(na_list)
                    confidence_list.extend(con_list)
                    box_bounds_list.extend(box_list)

            source_image_list.extend([os.path.basename(image1)]*len(box_bounds_list))

        if len(box_bounds_list) < 1:
            print('Warning, no predicted boxes in %s' % img_save_dir)
            return None

        # apply non_max_suppression
        # print('box_bounds_list',box_bounds_list)
        # print('confidence_list',confidence_list)
        pick_index = non_max_suppression(np.array(box_bounds_list), probs=np.array(confidence_list),
                                         overlapThresh=0.5,b_geo=True)
        # print('pick_index', pick_index)
        class_id_list = [class_id_list[idx] for idx in pick_index]
        name_list = [name_list[idx] for idx in pick_index ]
        confidence_list = [confidence_list[idx] for idx in pick_index ]
        box_bounds_list = [ box_bounds_list[idx] for idx in pick_index ]
        # to polygon
        box_poly_list = [ vector_gpd.convert_image_bound_to_shapely_polygon(item) for item in box_bounds_list]

        # box_poly_list

        # save to shapefile
        detect_boxes_dict = {'class_id':class_id_list, 'name':name_list, 'source_img':source_image_list,
                             'confidence':confidence_list, "Polygon":box_poly_list}
        save_pd = pd.DataFrame(detect_boxes_dict)
        ref_prj = map_projection.get_raster_or_vector_srs_info_proj4(image1)
        vector_gpd.save_polygons_to_files(save_pd,'Polygon',ref_prj,out_shp_path)


    return out_shp_path


def yolo_postProcess(para_file,inf_post_note,b_skip_getshp=False,test_id=None):
    # test_id is the related to training

    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    # the test string in 'exe.sh'
    test_note = inf_post_note

    WORK_DIR = os.getcwd()

    SECONDS = time.time()

    expr_name = parameters.get_string_parameters(para_file, 'expr_name')
    network_setting_ini = parameters.get_string_parameters(para_file,'network_setting_ini')


    inf_dir = parameters.get_directory(para_file, 'inf_output_dir')
    if test_id is None:
        test_id = os.path.basename(WORK_DIR) + '_' + expr_name

    # get name of inference areas
    multi_inf_regions = parameters.get_string_list_parameters(para_file, 'inference_regions')

    # run post-processing parallel
    # max_parallel_postProc_task = 8

    backup_dir = os.path.join(WORK_DIR, 'result_backup')
    io_function.mkdir(backup_dir)

    # loop each inference regions
    sub_tasks = []
    same_area_time_inis =  group_same_area_time_observations(multi_inf_regions)
    region_eva_reports = {}
    for key in same_area_time_inis.keys():
        multi_observations = same_area_time_inis[key]
        area_name = parameters.get_string_parameters(multi_observations[0], 'area_name')  # they have the same name and time
        area_time = parameters.get_string_parameters(multi_observations[0], 'area_time')
        merged_shp_list = []
        for area_idx, area_ini in enumerate(multi_observations):
            area_remark = parameters.get_string_parameters(area_ini, 'area_remark')
            area_save_dir, shp_pre,_ = get_observation_save_dir_shp_pre(inf_dir,area_name,area_time,area_remark,test_id)

            # get image list
            inf_image_dir = parameters.get_directory(area_ini, 'inf_image_dir')
            # it is ok consider a file name as pattern and pass it the following functions to get file list
            inf_image_or_pattern = parameters.get_string_parameters(area_ini, 'inf_image_or_pattern')
            inf_img_list = io_function.get_file_list_by_pattern(inf_image_dir,inf_image_or_pattern)
            img_count = len(inf_img_list)
            if img_count < 1:
                raise ValueError('No image for inference, please check inf_image_dir and inf_image_or_pattern in %s'%area_ini)

            merged_shp = os.path.join(WORK_DIR, area_save_dir, shp_pre + '.shp')
            if b_skip_getshp:
                pass
            else:
                # post image one by one
                result_shp_list = []
                for img_idx, img_path in enumerate(inf_img_list):
                    out_shp = yolo_results_to_shapefile(WORK_DIR, img_idx, area_save_dir, test_id)
                    if out_shp is not None:
                        result_shp_list.append(os.path.join(WORK_DIR,out_shp))
                # merge shapefiles
                merge_shape_files(result_shp_list,merged_shp)

            merged_shp_list.append(merged_shp)

        if b_skip_getshp is False:
            # add occurrence to each polygons
            get_occurence_for_multi_observation(merged_shp_list)

        for area_idx, area_ini in enumerate(multi_observations):
            area_remark = parameters.get_string_parameters(area_ini, 'area_remark')
            area_save_dir, shp_pre, area_remark_time  = get_observation_save_dir_shp_pre(inf_dir, area_name, area_time, area_remark,test_id)

            merged_shp = os.path.join(WORK_DIR, area_save_dir, shp_pre + '.shp')

            # add attributes to shapefile (no other attribute to add)
            # shp_attributes = os.path.join(WORK_DIR,area_save_dir, shp_pre+'_post_NOrm.shp')
            shp_attributes = merged_shp
            # add_polygon_attributes(merged_shp, shp_attributes, para_file, area_ini)

            # remove polygons
            shp_post = os.path.join(WORK_DIR, area_save_dir, shp_pre+'_post.shp')
            remove_polygons_main(shp_attributes, shp_post, para_file)

            # evaluate the mapping results
            out_report = os.path.join(WORK_DIR, area_save_dir, shp_pre+'_evaluation_report.txt')
            evaluation_polygons(shp_post,para_file,area_ini,out_report)

            ##### copy and backup files ######
            # copy files to result_backup
            if len(test_note) > 0:
                backup_dir_area = os.path.join(backup_dir, area_name + '_' + area_remark_time + '_' + test_id + '_' + test_note)
            else:
                backup_dir_area = os.path.join(backup_dir, area_name + '_' + area_remark_time + '_' + test_id )
            io_function.mkdir(backup_dir_area)
            if len(test_note) > 0:
                bak_merged_shp = os.path.join(backup_dir_area, '_'.join([shp_pre,test_note]) + '.shp')
                bak_post_shp = os.path.join(backup_dir_area, '_'.join([shp_pre,'post',test_note]) + '.shp')
                bak_eva_report = os.path.join(backup_dir_area, '_'.join([shp_pre,'eva_report',test_note]) + '.txt')
                bak_area_ini = os.path.join(backup_dir_area, '_'.join([shp_pre,'region',test_note]) + '.ini')
            else:
                bak_merged_shp = os.path.join(backup_dir_area, '_'.join([shp_pre]) + '.shp')
                bak_post_shp = os.path.join(backup_dir_area, '_'.join([shp_pre, 'post']) + '.shp')
                bak_eva_report = os.path.join(backup_dir_area, '_'.join([shp_pre, 'eva_report']) + '.txt')
                bak_area_ini = os.path.join(backup_dir_area, '_'.join([shp_pre, 'region']) + '.ini')

            io_function.copy_shape_file(merged_shp,bak_merged_shp)
            io_function.copy_shape_file(shp_post, bak_post_shp)
            io_function.copy_file_to_dst(area_ini, bak_area_ini, overwrite=True)

            if os.path.isfile(out_report):
                io_function.copy_file_to_dst(out_report, bak_eva_report, overwrite=True)
                region_eva_reports[shp_pre] = bak_eva_report



    if len(test_note) > 0:
        bak_para_ini = os.path.join(backup_dir, '_'.join([test_id,'para',test_note]) + '.ini' )
        # bak_network_ini = os.path.join(backup_dir, '_'.join([test_id,'network',test_note]) + '.ini' )
        bak_time_cost = os.path.join(backup_dir, '_'.join([test_id,'time_cost',test_note]) + '.txt' )
    else:
        bak_para_ini = os.path.join(backup_dir, '_'.join([test_id, 'para']) + '.ini')
        # bak_network_ini = os.path.join(backup_dir, '_'.join([test_id, 'network']) + '.ini')
        bak_time_cost = os.path.join(backup_dir, '_'.join([test_id, 'time_cost']) + '.txt')
    io_function.copy_file_to_dst(para_file, bak_para_ini)
    # io_function.copy_file_to_dst(network_setting_ini, bak_network_ini)
    if os.path.isfile('time_cost.txt'):
        io_function.copy_file_to_dst('time_cost.txt', bak_time_cost)

    # output the evaluation report to screen
    for key in region_eva_reports.keys():
        report = region_eva_reports[key]
        print('evaluation report for %s:'%key)
        os.system('head -n 7 %s'%report)

    # output evaluation report to table
    if len(test_note) > 0:
        out_table = os.path.join(backup_dir, '_'.join([test_id,'accuracy_table',test_note]) + '.xlsx' )
    else:
        out_table = os.path.join(backup_dir, '_'.join([test_id, 'accuracy_table']) + '.xlsx')
    eva_reports = [ region_eva_reports[key] for key in region_eva_reports]
    if len(eva_reports) > 0:
        eva_report_to_tables.eva_reports_to_table(eva_reports, out_table)

    duration= time.time() - SECONDS
    os.system('echo "$(date): time cost of post-procesing: %.2f seconds">>time_cost.txt'%duration)



    pass

def main(options, args):

    print(" YOLO Post-processing ")

    para_file = args[0]
    # the test string in 'exe.sh'
    if len(args) > 1:
        test_note = args[1]
    else:
        test_note = ''

    yolo_postProcess(para_file,test_note)


if __name__ == '__main__':
    usage = "usage: %prog [options] para_file test_note"
    parser = OptionParser(usage=usage, version="1.0 2021-04-08")
    parser.description = 'Introduction: Post-processing of YOLO prediction results '

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)

