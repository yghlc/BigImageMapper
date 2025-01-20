#!/usr/bin/env python
# Filename: extract_results_from_multi_source.py 
"""
introduction: extract image classification results from multiple sources

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 13 May, 2024
"""

import os,sys
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
# import parameters
import basic_src.basic as basic
from datetime import datetime
import basic_src.io_function as io_function
import basic_src.map_projection as map_projection
import datasets.vector_gpd as vector_gpd
import parameters

# from get_organize_training_data import extract_sub_image_labels_one_region
import pandas as pd

import random


def extract_classification_result_from_multi_sources(in_shp_list, save_path, extract_class_id=1, occurrence=4):
    '''
    extract the results which detected as the target multiple time during the multiple observation
    :param in_shp_list: file list for the multiple observation
    :param save_path: save path
    :param extract_class_id: the ID the target class
    :param occurrence: occurrence of the target during the multiple observation
    :return:
    '''

    if os.path.isfile(save_path):
        print(datetime.now(), '%s already exists, skip' % save_path)
        return

    print(datetime.now(), 'Input shape file:', in_shp_list)
    poly_class_ids_all = 'poly_class_ids_all.json'

    if os.path.isfile(poly_class_ids_all):
        print(datetime.now(), '%s already exists, read them directly'%poly_class_ids_all)
        poly_class_ids = io_function.read_dict_from_txt_json(poly_class_ids_all)
    else:
        # read
        poly_class_ids = {}
        for shp in in_shp_list:
            print(datetime.now(), 'reading %s'%shp)
            polyIDs = vector_gpd.read_attribute_values_list(shp,'polyID')
            preClassIDs = vector_gpd.read_attribute_values_list(shp, 'preClassID')
            _ = [poly_class_ids.setdefault(pid, []).append(c_id) for pid, c_id in zip(polyIDs,preClassIDs)]

        # save and organize them
        io_function.save_dict_to_txt_json('poly_class_ids_all.json',poly_class_ids)

    extract_class_id_results(in_shp_list[0], poly_class_ids, save_path, extract_class_id=extract_class_id, occurrence=occurrence)



def extract_class_id_results(shp_path, poly_class_ids, save_path, extract_class_id=1, occurrence = 4):
    '''
    extract the results which detected  as the target multiple time
    :param shp_path: a shape file contains points and "polyID"
    :param poly_class_ids: dict containing predicting results
    :param save_path: the save path for the results
    :param extract_class_id: the target id
    :param occurrence: occurrence time
    :return:
    '''
    save_json = 'poly_class_ids_id%d_occurrence%d.json' % (extract_class_id, occurrence)
    if os.path.isfile(save_json):
        print(datetime.now(),'warning, %s exists, read it directly'%save_json)
        sel_poly_class_ids = io_function.read_dict_from_txt_json(save_json)
        sel_poly_ids = list(sel_poly_class_ids.keys())
        print(datetime.now(), 'read %d results' % len(sel_poly_ids))
    else:
        print(datetime.now(), 'extract results for class: %d' % extract_class_id)
        sel_poly_ids = [ key for key in poly_class_ids.keys() if poly_class_ids[key].count(extract_class_id) >= occurrence ]
        print(datetime.now(), 'select %d results'%len(sel_poly_ids))
        # print(sel_poly_ids[:10])
        sel_poly_class_ids = {key: poly_class_ids[key] for key in sel_poly_ids}
        io_function.save_dict_to_txt_json(save_json, sel_poly_class_ids)

    # read and save shapefile
    save_shp = save_path
    sel_poly_ids_int = [int(item) for item in sel_poly_ids]
    vector_gpd.save_shapefile_subset_as_valueInlist(shp_path,save_shp,'polyID',sel_poly_ids_int)
    print(datetime.now(), 'save to %s and %s' % (save_json, save_shp))

def test_extract_class_id_results():
    shp_path = 'arctic_huang2023_620grids_s2_rgb_2023-predicted_classID.shp'
    poly_class_ids = io_function.read_dict_from_txt_json('poly_class_ids.json')
    save_path = 'sel_result.shp'
    extract_class_id_results(shp_path, poly_class_ids, save_path, extract_class_id=1, occurrence=7)


def extract_images_for_one_region(area_ini, out_dir, in_shp):

    get_subImage_script = os.path.join(code_dir, 'datasets', 'get_subImages.py')

    area_name_remark_time = parameters.get_area_name_remark_time(area_ini)
    extract_img_dir = os.path.join(out_dir, area_name_remark_time + '_images')
    io_function.mkdir(extract_img_dir)
    extract_done_indicator = os.path.join(extract_img_dir, 'extract_image_using_vector.done')

    dstnodata = 0
    buffersize = 1
    process_num = 8
    rectangle_ext = '--rectangle'

    image_dir = parameters.get_directory(area_ini, 'inf_image_dir')  # inf_image_dir
    image_or_pattern = parameters.get_string_parameters(area_ini, 'inf_image_or_pattern')  # inf_image_or_pattern

    # extract_sub_image_labels_one_region(area_save_dir, para_file, area_ini, b_training=False, b_convert_label=False)

    command_string = get_subImage_script + ' -b ' + str(buffersize) + ' -e ' + image_or_pattern + \
                     ' -o ' + extract_img_dir + ' -n ' + str(dstnodata) + ' -p ' + str(process_num) \
                     + ' ' + rectangle_ext + ' --no_label_image '
    # if each_image_equal_size is not None:
    #     command_string += ' -s %s  ' % str(each_image_equal_size)
    command_string += in_shp + ' ' + image_dir
    if os.path.isfile(extract_done_indicator):
        basic.outputlogMessage('Warning, sub-images already been extracted, read them directly')
    else:
        basic.os_system_exit_code(command_string)



def extract_images_for_manu_check(merge_result_shp, res_shp_list, out_dir, sample_num = 300, repeat_idx=1):

    io_function.is_file_exist(merge_result_shp)
    if os.path.isdir(out_dir) is False:
        io_function.mkdir(out_dir)

    # find region ini files
    multi_inf_regions = []
    for res_shp in res_shp_list:
        res_dir = os.path.dirname(res_shp)
        ini_files = io_function.get_file_list_by_pattern(res_dir,'*.ini')
        if len(ini_files) < 1:
            raise ValueError('there is not area ini files in %s'%res_dir)
        if len(ini_files) > 1:
            raise ValueError('there are multiple area ini files in %s' % res_dir)
        multi_inf_regions.append(ini_files[0])

    # randomly select results
    polys = vector_gpd.read_polygons_gpd(merge_result_shp,b_fix_invalid_polygon=False)
    index_list = list(range(len(polys)))
    if sample_num > len(polys):
        print('Warning, the set select count (%d) is greater than the total count of results (%d), select all'%(sample_num, len(polys)))
        sample_num = len(polys)
    sel_index = random.sample(index_list, sample_num)

    sel_merge_result_shp = os.path.join(out_dir,
        os.path.basename(io_function.get_name_by_adding_tail(merge_result_shp, 'R%d_random%d' % (repeat_idx,sample_num))))
    if os.path.isfile(sel_merge_result_shp):
        print('warning, %s exists, skip sampling'%sel_merge_result_shp)
    else:
        vector_gpd.save_shapefile_subset_as(sel_index,merge_result_shp, sel_merge_result_shp)

    # buffer to the same size
    buffer_size = 500
    sel_merge_result_shp_buff = io_function.get_name_by_adding_tail(sel_merge_result_shp,'buffer%d'%buffer_size)
    points, polyIDs = vector_gpd.read_polygons_attributes_list(sel_merge_result_shp,'polyID')
    polys = [item.buffer(buffer_size) for item in  points]
    data_pd = pd.DataFrame({'Polygons':polys, 'polyID':polyIDs})
    wkt = map_projection.get_raster_or_vector_srs_info_wkt(sel_merge_result_shp)
    vector_gpd.save_polygons_to_files(data_pd,'Polygons', wkt, sel_merge_result_shp_buff)


    # extract images for each region.
    for reg_ini in multi_inf_regions:
        extract_images_for_one_region(reg_ini,out_dir,sel_merge_result_shp_buff)



def main(options, args):

    res_shp_list = args
    res_shp_list = [ os.path.abspath(item) for item in res_shp_list]
    save_path = options.save_path
    target_id = options.target_id
    min_occurrence = options.occurrence
    sample_count = options.sample_count
    repeat_times = options.repeat_times

    # datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    datetime_str = datetime.now().strftime('%m%d%H') # only include month, day, hour
    if save_path is None:
        save_path = 'classID%d_occurrence%d_%s.shp'%(target_id,min_occurrence,datetime_str)


    extract_classification_result_from_multi_sources(res_shp_list, save_path,
                                                     extract_class_id = target_id,occurrence=min_occurrence)

    # sample_count = 300
    for repeat in range(repeat_times):
        extract_img_dir = (io_function.get_name_no_ext(save_path) +
                           '_R%d_%dsample_Imgs'%(repeat+1, sample_count))
        extract_images_for_manu_check(save_path,res_shp_list,extract_img_dir,sample_num=sample_count)


if __name__ == '__main__':

    # test_extract_class_id_results()
    # sys.exit(0)

    usage = "usage: %prog [options] res_shp1 res_shp2 res_shp3 ...  "
    parser = OptionParser(usage=usage, version="1.0 2024-05-13")
    parser.description = 'Introduction: extract image classification results from multiple sources '

    parser.add_option("-s", "--save_path",
                      action="store", dest="save_path",
                      help="the file path for saving the results")

    parser.add_option("-i", "--target_id",
                      action="store", dest="target_id", type=int, default=1,
                      help="the class id want to save")

    parser.add_option("-m", "--occurrence",
                      action="store", dest="occurrence", type=int, default=4,
                      help="minimum of the target ID in multiple observations ")

    parser.add_option("-c", "--sample_count",
                      action="store", dest="sample_count", type=int, default=300,
                      help="the total sample count to save")

    parser.add_option("-r", "--repeat_times",
                      action="store", dest="repeat_times", type=int, default=5,
                      help="the times to repeating the save random samples for validation")



    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)



    main(options, args)
