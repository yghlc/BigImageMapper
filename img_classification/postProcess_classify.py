#!/usr/bin/env python
# Filename: classify_postProcess.py 
"""
introduction: post-processing of the image classification results, especially for data without ground truth

1. write the top1 results into the original shapefile is possilble
2. randomly select some classes and sample for manually checking


authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 03 May, 2024
"""

import os,sys
from optparse import OptionParser


code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters
import basic_src.basic as basic
import basic_src.io_function as io_function
import basic_src.map_projection as map_projection
import datasets.vector_gpd as vector_gpd

# from prediction_clip import prepare_dataset
from get_organize_training_data import read_label_ids_local

import time
import random

import re
import pandas as pd

def get_class_name(class_id, class_id_dict):
    res = [ key for key in class_id_dict.keys() if class_id_dict[key] == class_id]
    if len(res) < 1:
        raise ValueError('class id: %d not in the dict: \n %s'%(class_id, str(class_id_dict)))
    return res[0]


def select_sample_for_manu_check(class_id, save_dir, sel_count, class_id_dict, image_path_list, res_dict):
    # image_base_name, confidence
    top1_predict = [ [key, res_dict[key]['confidence'][0]] for key in res_dict.keys() if res_dict[key]['pre_labels'][0] == class_id ]

    if sel_count > len(top1_predict):
        basic.outputlogMessage('warning, for class %d, select sample count (%d) is larger than the total number (%d) of all samples, '
                               'will choose all the samples' % (
            class_id, sel_count, len(top1_predict)))
        sel_top1_predict = top1_predict
    else:
        sel_top1_predict = random.sample(top1_predict, sel_count)

    top1_confidence = [item[1] for item in sel_top1_predict]
    top1_image_name = [item[0] for item in sel_top1_predict]
    top1_image_path = [item for item in image_path_list if os.path.basename(item) in top1_image_name]

    # save
    class_name = get_class_name(class_id, class_id_dict)
    outpur_dir = os.path.join(save_dir, 'id%d_%s_random_%d_samples'%(class_id, class_name, len(top1_image_name)))
    # remove it if already exist, then save new randomly saved one
    if os.path.isdir(outpur_dir):
        io_function.delete_file_or_dir(outpur_dir)
    io_function.mkdir(outpur_dir)

    # save the file name of all samples to txt
    top1_predict_all_txt = os.path.join(save_dir, 'id%d_%s_all_samples.txt'%(class_id, class_name))
    top1_predict_str = ['%s %f'%(item[0], item[1]) for item in top1_predict]
    io_function.save_list_to_txt(top1_predict_all_txt, top1_predict_str)

    # copy the file and save the confidences
    for image_path in top1_image_path:
        io_function.copyfiletodir(image_path,outpur_dir)
    top1_confidence_dict = {}
    for img_name, conf in zip(top1_image_name, top1_confidence):
        top1_confidence_dict[img_name] = conf
    save_json = outpur_dir + '_confidence.json'
    io_function.save_dict_to_txt_json(save_json,top1_confidence_dict)

def write_top1_result_into_vector_file(vector_path, res_dict, save_path, column_name='preClassID'):
    '''
    save the prediciton results (top 1) into vector file
    :param vector_path:
    :param res_dict: results in dict
    :return:
    '''
    # res_dict
    #         res_dict[os.path.basename(i_path)] = { }
    #         res_dict[os.path.basename(i_path)]['confidence'] = probs.tolist()
    #         res_dict[os.path.basename(i_path)]['pre_labels'] = labels.tolist()

    if os.path.isfile(save_path):
        print('Warning, %s already exists, skip'%save_path)
        return


    # for some case, a polygon may don't have the corresponding sub-images, then ignore it and set predict id as -1
    polys = vector_gpd.read_polygons_gpd(vector_path,b_fix_invalid_polygon=False)
    centroids = [ vector_gpd.get_polygon_centroid(item) for item in polys]

    predict_class_ids = [-1] * len(polys)

    # for a key: hillshade_HDLine_grid24872_14030.tif,  "14030" is the index of the polygon in the original shapefile (see get_subImages.py)
    for key in res_dict.keys():
        try:
            poly_idx = int(re.findall(r"_([0-9]+)\.", key)[0])
            predict_class_ids[poly_idx] = res_dict[key]['pre_labels'][0]
        except IndexError:
            print('IndexError found in dict %s, with key %s, poly_idx: %d, total count: %d '%(str(res_dict[key]), key, poly_idx, len(predict_class_ids)))
            print('count of polygons: %d'%len(polys))
        except:
            print('Other errors found in dict %s, with key %s, poly_idx: %d, total count: %d '%(str(res_dict[key]), key, poly_idx, len(predict_class_ids)))

    saved_attributes = {'Points': centroids, column_name:predict_class_ids}

    if vector_gpd.is_field_name_in_shp(vector_path,'polyID'):
        polyID_list = vector_gpd.read_attribute_values_list(vector_path,'polyID')
        saved_attributes['polyID'] = polyID_list
    else:
        saved_attributes['polyID'] = list(range(len(polys)))

    wkt = map_projection.get_raster_or_vector_srs_info_wkt(vector_path)
    data_pd = pd.DataFrame(saved_attributes)
    vector_gpd.save_points_to_file(data_pd,'Points',wkt,save_path)

def merge_result_from_multi_small_regions(para_file,multi_inf_regions):
    expr_name = parameters.get_string_parameters(para_file, 'expr_name')
    res_dir = os.path.join(parameters.get_directory(para_file, 'inf_output_dir'), expr_name)

    if len(multi_inf_regions) < 1:
        print('Only one region, skip merging')
        return

    # got result list
    merge_save_dir = None
    shp_list = []
    count_each_class_txt_list = []

    for area_ini in multi_inf_regions:
        area_name_remark_time = parameters.get_area_name_remark_time(area_ini)
        area_save_dir = os.path.join(res_dir, area_name_remark_time)
        sub_str = re.findall(r"_sub([0-9]+)_", area_name_remark_time)
        if len(sub_str)< 1:
            basic.outputlogMessage('warning, the folder name dont contain "sub" string, and may not are the sub-region after division, skip merging ')
            return

        sub_id = int(sub_str[0])
        if merge_save_dir is None:
            merge_save_dir_name = area_name_remark_time.replace('_sub%d_'%sub_id,'_')
            merge_save_dir = os.path.join(res_dir, merge_save_dir_name)
            io_function.mkdir(merge_save_dir)

        shp = os.path.join(area_save_dir, area_name_remark_time + '-predicted_classID.shp')
        count_txt = os.path.join(area_save_dir, 'prediction_count_each_class.txt')
        shp_list.append(shp)
        count_each_class_txt_list.append(count_txt)

    # merge shp
    merge_shp = os.path.join(merge_save_dir, os.path.basename(merge_save_dir) + '-predicted_classID.shp' )
    vector_gpd.merge_vector_files(shp_list,merge_shp)

    # merge txt
    class_count = {}
    for txt in count_each_class_txt_list:
        txt_lines = io_function.read_list_from_txt(txt)
        for line in txt_lines:
            c_name, count = line.split(':')
            count = int(count.strip() )
            class_count.setdefault(c_name, []).append(count)
    save_count_txt = os.path.join(merge_save_dir, 'prediction_count_each_class.txt')
    with open(save_count_txt, 'w') as f_obj:
        for key in class_count:
            f_obj.writelines('%s: %d \n'%(key,sum(class_count[key])))


def postProcessing_one_region(area_idx, area_ini, para_file, area_save_dir):

    res_json_path = os.path.join(area_save_dir, os.path.basename(area_save_dir) + '-classify_results.json')
    if os.path.isfile(res_json_path) is False:
        basic.outputlogMessage('Warning, %s results (a json file) for %s does not exist, skip'%(res_json_path,area_ini))
        return False

    # inf_image_dir = parameters.get_directory(area_ini, 'inf_image_dir')
    # inf_image_or_pattern = parameters.get_string_parameters(area_ini, 'inf_image_or_pattern')
    # dataset = prepare_dataset(para_file, area_ini,area_save_dir,inf_image_dir, inf_image_or_pattern,
    #                              transform=None,test=True)
    # image_path_list = dataset.img_list
    #         res_dict[os.path.basename(i_path)] = { }
    #         res_dict[os.path.basename(i_path)]['confidence'] = probs.tolist()
    #         res_dict[os.path.basename(i_path)]['pre_labels'] = labels.tolist()
    res_dict = io_function.read_dict_from_txt_json(res_json_path)
    if res_dict is None:
        raise IOError(f'the size of file: {res_json_path} is zero, may need to re-run the prediction,'
                      f'please run "ls -lhS grid*/*-classify_results.json > tmp_json" to check these files')

    class_labels_txt = parameters.get_file_path_parameters(para_file,'class_labels')
    class_id_dict = read_label_ids_local(class_labels_txt)

    #  count for each class ids
    all_prediction_count_txt = os.path.join(area_save_dir, 'prediction_count_each_class.txt')
    total_count = 0
    if os.path.isfile(all_prediction_count_txt) is False:
        with open(all_prediction_count_txt, 'w') as f_obj:
            for key in class_id_dict.keys():
                c_id = class_id_dict[key]
                top1_predict_c = [ [key, res_dict[key]['confidence'][0]] for key in res_dict.keys() if res_dict[key]['pre_labels'][0] == c_id ]
                f_obj.writelines('%s (id%d) count: %d \n'%(key, c_id, len(top1_predict_c)))
                total_count += len(top1_predict_c)
            f_obj.writelines('total count: %d \n' % total_count)
    else:
        print(f'Warning, {all_prediction_count_txt} already exists')


    # select sample for checking
    # move selection of random samples into prediction step (because after prediciton, these images will be removed)

    # class_ids_for_manu_check = parameters.get_string_list_parameters(para_file,'class_ids_for_manu_check')
    # class_ids_for_manu_check = [ int(item) for item in class_ids_for_manu_check]
    # sel_count = parameters.get_digit_parameters(para_file,'sample_num_per_class','int')
    # for c_id in class_ids_for_manu_check:
    #     select_sample_for_manu_check(c_id, area_save_dir, sel_count, class_id_dict, image_path_list, res_dict)

    # write results into shapefile
    all_polygons_labels = parameters.get_file_path_parameters_None_if_absence(area_ini, 'all_polygons_labels')
    save_shp_path = parameters.get_area_name_remark_time(area_ini) + '-predicted_classID.shp'
    save_shp_path = os.path.join(area_save_dir, save_shp_path)
    if all_polygons_labels is not None:
        write_top1_result_into_vector_file(all_polygons_labels, res_dict, save_shp_path)



def postProcessing_main(para_file):
    print("post-Processing for image classification")
    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))
    SECONDS = time.time()

    expr_name = parameters.get_string_parameters(para_file, 'expr_name')

    res_dir = os.path.join(parameters.get_directory(para_file, 'inf_output_dir'), expr_name)
    multi_inf_regions = parameters.get_string_list_parameters(para_file, 'inference_regions')

    for area_idx, area_ini in enumerate(multi_inf_regions):

        area_name_remark_time = parameters.get_area_name_remark_time(area_ini)
        area_save_dir = os.path.join(res_dir, area_name_remark_time)
        io_function.mkdir(area_save_dir)

        # post processing
        print('%d/%d, post-processing for %s'%(area_idx, len(multi_inf_regions), area_ini))
        postProcessing_one_region(area_idx, area_ini, para_file, area_save_dir)


    # merge results for several regions (a large region that were divided into many small one using ./divide_to_small_region_ini.py )
    b_merge_results_from_regions = parameters.get_bool_parameters_None_if_absence(para_file,'b_merge_results_from_regions')
    if b_merge_results_from_regions is True:
        merge_result_from_multi_small_regions(para_file,multi_inf_regions)


    duration= time.time() - SECONDS
    os.system('echo "$(date): time cost of post-Processing for image classification: %.2f seconds">>time_cost.txt'%duration)


def main(options, args):

    para_file = args[0]
    postProcessing_main(para_file)


if __name__ == '__main__':
    usage = "usage: %prog [options] para_file"
    parser = OptionParser(usage=usage, version="1.0 2024-05-03")
    parser.description = 'Introduction: post-processing for the image classification classes '

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)


    main(options, args)
