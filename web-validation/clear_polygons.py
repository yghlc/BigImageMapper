#!/usr/bin/env python
# Filename: clear_polygons.py 
"""
introduction: filter the polygons after crowdsourcing validation

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 26 September, 2022
"""

import os,sys
from optparse import OptionParser

import pandas as pd
import re
import json

deeplabforRS = os.path.expanduser('~/codes/PycharmProjects/DeeplabforRS')
sys.path.insert(0, deeplabforRS)
import vector_gpd
import basic_src.io_function as io_function
import basic_src.basic as basic

# from vector_features import max_IoU_score
from vector_features import IoU

# import nms
import numpy as np

defined_possibility = ['yes', 'high', 'medium', 'low', 'no']

possib2value = {'yes':1.0, 'high':0.75, 'medium':0.5, 'low':0.25, 'no':0.0}

nms_threshold = 0.5
possibility_threshold = 0.5

# a dict of user name and ids
# user_names = {}     # user_id <-> user_name

# def read_user_names(user_json_file):
#     # read json files, to many dict
#     with open(user_json_file) as f_obj:
#         data_list = json.load(f_obj)
#
#     for rec in data_list:
#         user_names[rec['pk']] = rec['fields']['username']       # id : username


def non_max_suppression_polygons(polygons, scores,nms_iou_threshold=0.5):

    # alternative: https://nms.readthedocs.io/ (not able to make it works)

    # return indicies
    if len(polygons) != len(scores):
        raise ValueError('the length of polygons and scores are different')

    count = len(polygons)
    all_idx = [ item for item in range(count)]
    rm_idx = []
    # keep_idx = []
    # iou_array = np.zeros((count,count),dtype=np.float32)
    for i in range(count-1):
        if i in rm_idx:
            continue

        for j in range(i+1, count):
            if j in rm_idx:
                continue

            iou = IoU(polygons[i],polygons[j])

            if iou >= nms_iou_threshold:
                if scores[i] >= scores[j]:
                    # keep_idx.append(i)
                    rm_idx.append(j)
                else:
                    # keep_idx.append(j)
                    rm_idx.append(i)
                    break

    return list(set(all_idx).difference(set(rm_idx)))


def read_a_geojson_latlon(file_path):
    polygons = vector_gpd.read_shape_gpd_to_NewPrj(file_path,'EPSG:4326')
    # print(polygons)
    # only keep the one with Polygon type
    check_polygons = []
    for poly in polygons:
        # print(poly.type,type(poly.type))
        if poly.type == 'Polygon':
            check_polygons.append(poly)
        else:
            basic.outputlogMessage('Warning, ignore one geometry which is not Polygon but %s in %s '%(poly.type, file_path))

    return check_polygons


def test_read_a_geojson_latlon():
    # geojson = '/Users/huanglingcao/Data/labelearth.colorado.edu/data/data/objectPolygons/img000001_panArctic_time0_poly_3028.geojson'
    # geojson = '/Users/huanglingcao/Data/labelearth.colorado.edu/data/data/objectPolygons/img000001_panArctic_time0_poly_3028_by_CarolXU.geojson'

    # geojson = '/Users/huanglingcao/Data/labelearth.colorado.edu/data/data/objectPolygons/img000010_panArctic_time0_poly_432_by_liulin@cuhk.edu.hk_000.geojson'
    # read_a_geojson_latlon(geojson)
    pass

def merge_inputs_from_users(userinput_json_file,dir_geojson,user_json_file,image_json,save_path):
    with open(userinput_json_file) as f_obj:
        userinput_list = json.load(f_obj)

    with open(user_json_file) as f_obj:
        user_list = json.load(f_obj)

    with open(image_json) as f_obj:
        image_json = json.load(f_obj)

    user_names = {}
    for rec in user_list:
        user_names[rec['pk']] = rec['fields']['username']       # id : username

    image_polygon_file = {}
    for rec in image_json:
        image_polygon_file[rec['pk']] = rec['fields']['image_object_path']       # image id : path to bounding boxes

    image_polygons_valid_res = {}
    for rec in userinput_list:
        image_polygons_valid_res.setdefault('input_pk', []).append(rec['pk'])
        image_polygons_valid_res.setdefault('image_pk', []).append(rec['fields']['image_name'])
        image_polygons_valid_res.setdefault('user_pk', []).append(rec['fields']['user_name'])
        org_json = os.path.basename(image_polygon_file[rec['fields']['image_name']])
        image_polygons_valid_res.setdefault('org_polygon', []).append(org_json)    # original polygons
        image_polygons_valid_res.setdefault('user_name', []).append(user_names[ rec['fields']['user_name']])
        image_polygons_valid_res.setdefault('possibility', []).append(rec['fields']['possibility'])
        image_polygons_valid_res.setdefault('user_note', []).append(rec['fields']['user_note'])
        image_polygons_valid_res.setdefault('user_polygon', []).append(rec['fields']['user_image_output'])          # modified or added polygons


    ## save to excel file
    save_xlsx = io_function.get_name_no_ext(save_path) + '_all_records.xlsx'
    table_pd = pd.DataFrame(image_polygons_valid_res)
    with pd.ExcelWriter(save_xlsx) as writer:
        table_pd.to_excel(writer)
        print('saving to %s'%os.path.abspath(save_xlsx))


    # convert and save to shapefile
    user_name_list = []
    possibility_list = []
    org_polygon_name_list = []
    user_note_list = []
    polygon_list = []

    for user_name, possibility,user_note, org_json,user_input_json in zip(image_polygons_valid_res['user_name'],
                                                                          image_polygons_valid_res['possibility'],
                                                                          image_polygons_valid_res['user_note'],
                                                                          image_polygons_valid_res['org_polygon'],
                                                                          image_polygons_valid_res['user_polygon']):
        # add original polygons
        # print(org_json)
        org_json_path = os.path.join(dir_geojson, org_json)
        polygon_list.append(read_a_geojson_latlon(org_json_path)[0])
        user_name_list.append(user_name)
        org_polygon_name_list.append(org_json)
        possibility_list.append(possibility)
        user_note_list.append(user_note)

        if user_input_json is not None:
            # print(user_input_json)
            user_input_json_path = os.path.join(dir_geojson, os.path.basename(user_input_json) )
            new_polys = read_a_geojson_latlon(user_input_json_path)
            for n_poly in new_polys:
                polygon_list.append(n_poly)
                user_name_list.append(user_name)
                org_polygon_name_list.append(org_json)
                possibility_list.append('useradd')      # these are addded by users
                user_note_list.append(user_note)    # could be duplicated, but ok

    # save to file
    polygon_pd = pd.DataFrame({'user':user_name_list,'o_geojson':org_polygon_name_list, 'possibilit':possibility_list,
                               'note':user_note_list,'polygons':polygon_list})
    vector_gpd.save_polygons_to_files(polygon_pd, 'polygons', 'EPSG:4326', save_path,format='ESRI Shapefile')
    print('saving to %s' % os.path.abspath(save_path))


def handle_original_polygons(poly_index,all_polygons,all_possibilities):

    # return index
    save_poly_idx = None

    ori_poly_possi = []
    ori_poly_idx = []
    useradd_poly_idx = []
    for idx in poly_index:
        if all_possibilities[idx] in defined_possibility:
            ori_poly_possi.append(all_possibilities[idx])
            ori_poly_idx.append(idx)
        else:
            useradd_poly_idx.append(idx)    # useradd
    #
    ori_poly_possi_value = [ possib2value[item] for item in ori_poly_possi ]
    valid_count = len(ori_poly_possi_value)
    avg_possi = sum(ori_poly_possi_value)/valid_count
    if avg_possi >= possibility_threshold:
        # # check modified polygons in "useradd_poly_idx"
        # ***no need to check this, eventually, we will check all overlap polygons***

        # org_polygon = all_polygons[ori_poly_idx[0]] # may have multiple polygon, but they are the same
        # user_modify_added_polygons = [ all_polygons[idx] for idx in useradd_poly_idx]
        # iou_value = max_IoU_score(org_polygon, user_modify_added_polygons)
        # if iou_value > 0:
        #     # do something
        #    pass
        save_poly_idx = ori_poly_idx[0]  # may have multiple polygon, but they are the same

    return save_poly_idx, avg_possi,valid_count, useradd_poly_idx

def filter_polygons_based_on_userInput(all_polygon_shp,save_path):

    # read polygons
    all_polygons, possibilities = vector_gpd.read_polygons_attributes_list(all_polygon_shp,'possibilit',b_fix_invalid_polygon=False)
    org_geojson = vector_gpd.read_attribute_values_list(all_polygon_shp,'o_geojson')
    # users = vector_gpd.read_attribute_values_list(all_polygon_shp,'user')
    notes = vector_gpd.read_attribute_values_list(all_polygon_shp,'note')

    # remove ones with possibility as "NULL" (None); remove some copied ones
    new_all_polygons=[]
    new_possibilities = []
    new_org_geojson=[]
    rm_None_possi = 0
    rm_copied_input = 0
    for poly,p_text, o_json, user_note in zip(all_polygons,possibilities,org_geojson,notes):
        if p_text is None:
            rm_None_possi += 1
            continue
        if user_note == 'copy from lingcao.huang@colorado.edu':
            rm_copied_input += 1
            continue
        new_all_polygons.append(poly)
        new_possibilities.append(p_text)
        new_org_geojson.append(o_json)
    print('remove %d records of user input, with a possibility of None'%rm_None_possi)
    print('remove %d records of copied user input'%rm_copied_input)
    all_polygons = new_all_polygons
    possibilities = new_possibilities
    org_geojson = new_org_geojson


    save_polygon_list = []
    save_val_time_list = []
    save_possi_list = []    # keep
    # save_user_list = []
    # save_note_list = []
    # save_org_geojson_list = []

    all_modify_add_poly_idx_list = []

    img_possi_index = {}
    for idx, img_geojson in enumerate(org_geojson):
        if img_geojson in img_possi_index.keys():
            img_possi_index[img_geojson].append(idx)
        else:
            img_possi_index[img_geojson] = [idx]

    # print('\n count of unique_img_geojson:', len(img_possi_index.keys()))
    for img_geojson in img_possi_index.keys():
        poly_index = img_possi_index[img_geojson]

        # processing original boxes, keep or drop
        # one_poly_possi = [ possibilities[item] for item in poly_index if possibilities[item] in defined_possibility ]
        # print(one_poly_possi)

        save_poly_idx, avg_possi, valid_times, modify_add_poly_idx_list = handle_original_polygons(poly_index,all_polygons,possibilities)
        if save_poly_idx is not None:
            save_polygon_list.append(all_polygons[save_poly_idx])
            save_val_time_list.append(valid_times)
            save_possi_list.append(avg_possi)

        all_modify_add_poly_idx_list.extend(modify_add_poly_idx_list)

        # if img_geojson == 'img000001_panArctic_time0_poly_3028.geojson':
        #     break
        #
        # pass

    print('number of saved polygons from original boxes:', len(save_polygon_list))
    print('user added or modified polygons:', len(all_modify_add_poly_idx_list))

    # statistics before non_max_suppression
    print('statistics of the saved polygons from original boxes before non_max_suppression:')
    # possi_count_per = {}
    for p_value in sorted(set(save_possi_list),reverse=True):   # descending
        p_value_count = save_possi_list.count(p_value)
        print('Possibility: %lf, count: %d, percent: %lf ' % (p_value,p_value_count, p_value_count/len(save_polygon_list)))
        # possi_count_per[p_value] = save_possi_list.count(p_value)

    # for the polygons added by users keep for some of them, overlap each other, only keep one
    all_modify_add_polygons = [all_polygons[item] for item in all_modify_add_poly_idx_list ]
    all_modify_add_poly_scores = [1.0]*len(all_modify_add_polygons)

    save_polygon_list.extend(all_modify_add_polygons)
    save_val_time_list.extend([0]*len(all_modify_add_polygons))

    scores = [ item - 0.01 for item in save_possi_list]     # make the score of original polygon smaller, so priority to user modified polygons
    scores.extend(all_modify_add_poly_scores)

    save_possi_list.extend([1] * len(all_modify_add_polygons))

    print('Before non_max_suppression: %d polygons (polygons from original boxes and user added/modified ones):' % len(save_polygon_list))

    keep_idx = non_max_suppression_polygons(save_polygon_list,scores,nms_iou_threshold=nms_threshold)

    # save to file
    final_polys = [save_polygon_list[idx] for idx in keep_idx]
    final_val_times = [save_val_time_list[idx] for idx in keep_idx]
    final_possis = [save_possi_list[idx] for idx in keep_idx]
    print('After non_max_suppression, keep %d polygons:'%len(final_polys))

    # save to file
    polygon_pd = pd.DataFrame({'id':[i+1 for i in range(len(final_polys))],'possibilit':final_possis,
                               'val_times':final_val_times,'polygons': final_polys})
    vector_gpd.save_polygons_to_files(polygon_pd, 'polygons', 'EPSG:4326', save_path, format='ESRI Shapefile')
    print('saving to %s' % os.path.abspath(save_path))


def test_filter_polygons_based_on_userInput():
    all_polygon_shp = '/Users/huanglingcao/Data/labelearth.colorado.edu/data/thawslump_boxes/pan_arctic_thawslump_after_webValidation_NoFilter.shp'
    save_path = 'filter_result.shp'

    filter_polygons_based_on_userInput(all_polygon_shp,save_path)

def statistics_of_user_input(before_filter_shp,after_filter_shp,save_txt):
    possibility_no_filter = vector_gpd.read_attribute_values_list(before_filter_shp, 'possibilit')
    number_poly_user_add_edit = possibility_no_filter.count('useradd')

    # after filtering
    possibility = vector_gpd.read_attribute_values_list(after_filter_shp, 'possibilit')
    val_times  = vector_gpd.read_attribute_values_list(after_filter_shp, 'val_times')

    possibility_noZero_val = [ p for p, vt in zip(possibility,val_times) if vt > 0 ]
    possibility_zero_val = [ p for p, vt in zip(possibility,val_times) if vt == 0 ]

    possi_count_noZero_val = {}
    for p_value in sorted(set(possibility_noZero_val),reverse=True):   # descending
        print('get the number of the possibility: %s'%str(p_value))
        possi_count_noZero_val[p_value] = possibility_noZero_val.count(p_value)

    possi_count_zero_val = {}
    for p_value in sorted(set(possibility_zero_val),reverse=True):   # descending
        print('get the number of the possibility: %s'%str(p_value))
        possi_count_zero_val[p_value] = possibility_zero_val.count(p_value)

    # save to txt
    with open(save_txt,'w') as f_obj:
        f_obj.writelines('before merging and filtering: \n')
        f_obj.writelines('polygon count: %d \n'%len(possibility_no_filter))
        f_obj.writelines('count of polygons added or edited by users: %d \n'%number_poly_user_add_edit)

        f_obj.writelines('\nAfter merging and filtering: \n')
        count_noZero_val = len(possibility_noZero_val)
        count_zero_val = len(possibility_zero_val)
        f_obj.writelines('\nCount of polygons from original boxes: %d, count and percent for each possibility: \n'%count_noZero_val)
        for p_value in possi_count_noZero_val.keys():
            f_obj.writelines("Possibility: %lf, count: %d, percent: %.4lf \n"%(p_value, possi_count_noZero_val[p_value], possi_count_noZero_val[p_value]/count_noZero_val))

        f_obj.writelines('\nCount of polygons added or edited by users: %d, count and percent for each possibility:\n'%count_zero_val)
        for p_value in possi_count_zero_val.keys():
            f_obj.writelines("Possibility: %lf, count: %d, percent: %.4lf"%(p_value, possi_count_zero_val[p_value], possi_count_zero_val[p_value]/count_zero_val))

    print('saved to %s'%os.path.abspath(save_txt))



def main(options, args):
    userinput_json = args[0]
    dir_geojson = args[1]
    save_path = options.save_path
    user_json = options.user_json
    image_json = options.image_json

    io_function.is_file_exist(userinput_json)
    io_function.is_folder_exist(dir_geojson)
    if save_path is None:
        save_path = 'polygons_after_webValidation.shp'

    print('nms_threshold: %f, possibility_threshold: %f'%(nms_threshold,possibility_threshold))

    # merge polygons
    before_filter_save = io_function.get_name_by_adding_tail(save_path,'NoFilter')
    merge_inputs_from_users(userinput_json,dir_geojson, user_json,image_json,before_filter_save)

    # filter polygons
    filter_polygons_based_on_userInput(before_filter_save, save_path)

    save_txt = os.path.splitext(save_path)[0] + '_statistics.txt'
    statistics_of_user_input(before_filter_save, save_path, save_txt)



if __name__ == '__main__':
    usage = "usage: %prog [options] userinput.json dir_objectPolygons"
    parser = OptionParser(usage=usage, version="1.0 2022-09-26")
    parser.description = 'Introduction: clear polygons after web-based validation '

    parser.add_option("-s", "--save_path",
                      action="store", dest="save_path",
                      help="the path for saving file")

    parser.add_option("-u", "--user_json",
                      action="store", dest="user_json",
                      help="the json file containing user information, exported by Django")

    parser.add_option("-i", "--image_json",
                      action="store", dest="image_json",
                      help="the json file containing image information, exported by Django")


    (options, args) = parser.parse_args()
    # print(options.create_mosaic)

    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)
