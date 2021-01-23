#!/usr/bin/env python
# Filename: postProcess.py 
"""
introduction: convert inference results to shapefile, removing false some faslse postives, and so on

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 22 January, 2021
"""
import os, sys
import time

def inf_results_to_shapefile(curr_dir,img_idx, area_save_dir, test_id):

    img_save_dir = os.path.join(area_save_dir, 'I%d' % img_idx)
    out_name = os.path.basename(area_save_dir) + '_' + test_id

    os.chdir(img_save_dir)

    merged_tif = 'I%d'%img_idx + '_' + out_name + '.tif'
    if os.path.isfile(merged_tif):
        print('%s already exist'%merged_tif)
    else:
        #gdal_merge.py -init 0 -n 0 -a_nodata 0 -o I${n}_${output} I0_*.tif
        command_string = 'gdal_merge.py  -init 0 -n 0 -a_nodata 0 -o ' + merged_tif + ' I0_*.tif'
        res  = os.system(command_string)
        if res != 0:
            sys.exit(res)

    # to shapefile
    out_shp = 'I%d'%img_idx + '_' + out_name + '.shp'
    if os.path.isfile(out_shp):
        print('%s already exist' % out_shp)
    else:
        command_string = 'gdal_polygonize.py -8 %s -b 1 -f "ESRI Shapefile" %s'%(merged_tif,out_shp)
        res  = os.system(command_string)
        if res != 0:
            sys.exit(res)

    os.chdir(curr_dir)
    out_shp_path = os.path.join(img_save_dir,out_shp)
    return out_shp_path

def add_polygon_attributes(script, in_shp_path, save_shp_path, para_file, data_para_file):

    command_string = script +' -p %s -d %s %s %s' % (para_file,data_para_file, in_shp_path, save_shp_path)
    # print(command_string)
    res = os.system(command_string)
    print(res)
    if res != 0:
        sys.exit(res)


def remove_polygons(script, in_shp_path, save_shp_path, para_file):

    command_string = script + ' -p %s -o %s %s' % (para_file, save_shp_path, in_shp_path)
    res = os.system(command_string)
    if res != 0:
        sys.exit(res)

def evaluation_polygons(script, in_shp_path, para_file):

    command_string = script + ' -p %s %s' % (para_file, in_shp_path)
    res = os.system(command_string)
    if res != 0:
        sys.exit(res)
    return in_shp_path


if __name__ == '__main__':
    print("%s : Post-processing" % os.path.basename(sys.argv[0]))

    para_file = sys.argv[1]
    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    if len(sys.argv) > 2:
        post_note = sys.argv[2]
    else:
        post_note = ''

    code_dir = os.path.join(os.path.dirname(sys.argv[0]), '..')
    sys.path.insert(0, code_dir)
    import parameters
    import basic_src.io_function as io_function

    sys.path.insert(0, os.path.join(code_dir,'datasets'))
    from merge_shapefiles import merge_shape_files


    WORK_DIR = os.getcwd()

    expr_name = parameters.get_string_parameters(para_file, 'expr_name')
    network_setting_ini = parameters.get_string_parameters(para_file,'network_setting_ini')


    inf_dir = parameters.get_directory(para_file, 'inf_output_dir')
    test_id = os.path.basename(WORK_DIR) + '_' + expr_name

    # get name of inference areas
    multi_inf_regions = parameters.get_string_list_parameters(para_file, 'inference_regions')

    # run post-processing parallel
    # max_parallel_postProc_task = 8

    # loop each inference regions
    sub_tasks = []
    for area_idx, area_ini in enumerate(multi_inf_regions):
        area_name = parameters.get_string_parameters_None_if_absence(area_ini, 'area_name')
        area_remark = parameters.get_string_parameters_None_if_absence(area_ini, 'area_remark')

        area_save_dir = os.path.join(inf_dir, area_name + '_' + area_remark)

        # get image list
        inf_image_dir = parameters.get_directory(area_ini, 'inf_image_dir')
        # it is ok consider a file name as pattern and pass it the following functions to get file list
        inf_image_or_pattern = parameters.get_string_parameters(area_ini, 'inf_image_or_pattern')
        inf_img_list = io_function.get_file_list_by_pattern(inf_image_dir,inf_image_or_pattern)
        img_count = len(inf_img_list)
        if img_count < 1:
            raise ValueError('No image for inference, please check inf_image_dir and inf_image_or_pattern in %s'%area_ini)

        # post image one by one
        result_shp_list = []
        for img_idx, img_path in enumerate(inf_img_list):
            out_shp = inf_results_to_shapefile(WORK_DIR, img_idx, area_save_dir, test_id)
            result_shp_list.append(out_shp)

        # merge shapefiles
        shp_pre = os.path.basename(area_save_dir) + '_' + test_id
        merged_shp =  os.path.join(area_save_dir, shp_pre + '.shp')
        merge_shape_files(result_shp_list,merged_shp)

        # add attributes to shapefile
        add_attributes_script = os.path.join(code_dir,'datasets', 'get_polygon_attributes.py')
        shp_attributes = os.path.join(area_save_dir, shp_pre+'_post_NOrm.shp')
        add_polygon_attributes(add_attributes_script,merged_shp, shp_attributes, para_file, area_ini )

        # remove polygons
        rm_polygon_script = os.path.join(code_dir,'datasets', 'remove_mappedPolygons.py')
        shp_removed = os.path.join(area_save_dir, shp_pre+'_post.shp')
        remove_polygons(rm_polygon_script,shp_attributes, shp_removed, para_file)

        # evaluate the mapping results
        eval_shp_script = os.path.join(deeplabRS, 'evaluation_result.py')
        evaluation_polygons(eval_shp_script, shp_removed,para_file)








