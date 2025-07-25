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
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters
import basic_src.map_projection as map_projection
import basic_src.io_function as io_function
import basic_src.basic as basic

sys.path.insert(0, os.path.join(code_dir, 'datasets'))
from merge_shapefiles import merge_shape_files
import utility.eva_report_to_tables as eva_report_to_tables

from datasets.get_polygon_attributes import add_polygon_attributes
from datasets.remove_mappedPolygons import remove_polygons_main
from datasets.evaluation_result import evaluation_polygons

import datasets.raster_io as raster_io
import datasets.vector_gpd as vector_gpd


def merge_polygon_rasterize(ref_raster, in_shp, out_shp=None, work_dir='./', b_rm_tif=True):
    # union polygons touch each other in a shapefile

    in_polygons = vector_gpd.read_polygons_gpd(in_shp,b_fix_invalid_polygon=False)

    # rasterize to raster
    save_raster = os.path.basename(io_function.get_name_by_adding_tail(ref_raster,'merge'))
    save_raster = os.path.join(work_dir,save_raster)
    raster_io.burn_polygons_to_a_raster(ref_raster,in_polygons,1,save_raster)

    # set nodata
    if raster_io.set_nodata_to_raster_metadata(save_raster,0) is False:
        raise IOError('Set nodata failed for %s'%save_raster)

    # polygonize
    out_shp = vector_gpd.raster2shapefile(save_raster, out_shp=out_shp, connect8=True,format='ESRI Shapefile')
    if out_shp is None:
        raise IOError('polygonzied failed for %s' % save_raster)

    if b_rm_tif:
        io_function.delete_file_or_dir(save_raster)

    return out_shp

def inf_results_to_shapefile(curr_dir,img_idx, area_save_dir, test_id):

    img_save_dir = os.path.join(area_save_dir, 'I%d' % img_idx)
    out_name = os.path.basename(area_save_dir) + '_' + test_id

    # in SAM prediction results, if no prompts for a region, then, no folder. Feb 13, 2025
    if os.path.isdir(img_save_dir):
        os.chdir(img_save_dir)
    else:
        basic.outputlogMessage(f'Warning, {img_save_dir} does not exists, NO inference results')
        return None, None

    merged_tif = 'I%d'%img_idx + '_' + out_name + '.tif'
    if os.path.isfile(merged_tif):
        print('%s already exist'%merged_tif)
    else:
        # check if tif exists
        tif_list = io_function.get_file_list_by_pattern('./', 'I0_*.tif')
        if len(tif_list) < 1:
            print('Warning, NO I0_*.tif in %s'%img_save_dir)
            os.chdir(curr_dir)
            return None, None

        #gdal_merge.py -init 0 -n 0 -a_nodata 0 -o I${n}_${output} I0_*.tif
        command_string = 'gdal_merge.py  -init 0 -n 0 -a_nodata 0 -o ' + merged_tif + ' I0_*.tif'
        res  = os.system(command_string)
        if res != 0:
            # sys.exit(1)
            os.chdir(curr_dir)
            return None, None

    # to shapefile
    out_shp = 'I%d'%img_idx + '_' + out_name + '.shp'
    if os.path.isfile(out_shp):
        print('%s already exist' % out_shp)
    else:
        command_string = 'gdal_polygonize.py -8 %s -b 1 -f "ESRI Shapefile" %s'%(merged_tif,out_shp)
        res  = os.system(command_string)
        if res != 0:
            sys.exit(1)

    os.chdir(curr_dir)
    out_shp_path = os.path.join(img_save_dir,out_shp)
    out_raster = os.path.join(img_save_dir,merged_tif)
    return out_shp_path, out_raster

def inf_results_gpkg_to_shapefile(curr_dir,img_idx, area_save_dir, test_id):
    # when the inference directly output vector files (gpkg format),
    # merge and convert them to shapefile
    img_save_dir = os.path.join(area_save_dir, 'I%d' % img_idx)
    out_name = os.path.basename(area_save_dir) + '_' + test_id

    # in SAM prediction results, if no prompts for a region, then, no folder. Feb 13, 2025
    if os.path.isdir(img_save_dir):
        os.chdir(img_save_dir)
    else:
        basic.outputlogMessage(f'Warning, {img_save_dir} does not exists, NO inference results')
        return None

    # get the path of ref_raster before "chdir", otherwise, has problem for relative path
    img_idx_txt = os.path.join('../', '%d.txt' % img_idx)
    # img_idx_txt = os.path.join(img_save_dir,'../', '%d.txt' % img_idx)
    ref_raster = io_function.read_list_from_txt(img_idx_txt)[0]
    ref_raster = os.path.abspath(ref_raster)
    height, width, band_num, date_type = raster_io.get_height_width_bandnum_dtype(ref_raster)

    # to shapefile
    out_shp = 'I%d' % img_idx + '_' + out_name + '.shp'
    if os.path.isfile(out_shp):
        print('%s already exist' % out_shp)
    else:
        command_string = 'ogrmerge.py -o %s -f "ESRI Shapefile"  -single -progress "I0_*.gpkg" ' % (out_shp)
        res = os.system(command_string)
        if res != 0:
            # sys.exit(1)
            os.chdir(curr_dir)
            return None

        if height*width > 50000*50000:
            basic.outputlogMessage(f'Warning, skip merging polygons touch each other using rasterize, '
                                   f'as the image is too large ({height} by {width}), please use Shapely if necessary')
        else:
            # in the shapefile, merge those polygons touch each other
            merge_poly_shp = io_function.get_name_by_adding_tail(out_shp,'merge')
            out_shp = merge_polygon_rasterize(ref_raster,out_shp,out_shp=merge_poly_shp)

    os.chdir(curr_dir)
    out_shp_path = os.path.join(img_save_dir, out_shp)
    return out_shp_path



# def add_polygon_attributes(script, in_shp_path, save_shp_path, para_file, data_para_file):
#
#     command_string = script +' -p %s -d %s %s %s' % (para_file,data_para_file, in_shp_path, save_shp_path)
#     # print(command_string)
#     res = os.system(command_string)
#     print(res)
#     if res != 0:
#         sys.exit(1)


# def remove_polygons(script, in_shp_path, save_shp_path, para_file):
#
#     command_string = script + ' -p %s -o %s %s' % (para_file, save_shp_path, in_shp_path)
#     res = os.system(command_string)
#     if res != 0:
#         sys.exit(1)

# def evaluation_polygons(script, in_shp_path, para_file, data_para_file,out_report):
#
#     command_string = script + ' -p %s -d %s -o %s %s' % (para_file, data_para_file, out_report, in_shp_path)
#     res = os.system(command_string)
#     if res != 0:
#         sys.exit(1)
#     return in_shp_path

def group_same_area_time_observations(arae_ini_files):
    # group the observation with the same area name and time.
    same_area_time_obs_ini = {}
    for area_ini in arae_ini_files:
        area_name = parameters.get_string_parameters(area_ini, 'area_name')
        area_time = parameters.get_string_parameters(area_ini, 'area_time')

        area_time_key = area_name + '-' + area_time     # use '-' instead of '_' ('_ has been used in many place')
        if area_time_key not in same_area_time_obs_ini.keys():
            same_area_time_obs_ini[area_time_key] = []
        same_area_time_obs_ini[area_time_key].append(area_ini)

    return same_area_time_obs_ini

def get_observation_save_dir_shp_pre(inf_dir, area_name, area_time, area_remark,train_id):
    area_remark_time = area_remark + '_' + area_time
    area_save_dir = os.path.join(inf_dir, area_name + '_' + area_remark_time)
    shp_pre = os.path.basename(area_save_dir) + '_' + train_id
    return area_save_dir, shp_pre, area_remark_time


def get_occurence_for_multi_observation(shp_list):
    if len(shp_list) < 2:
        return False

    # need for calculating the occurrence.
    cd_dir = os.path.expanduser('~/codes/PycharmProjects/ChangeDet_DL/thawSlumpChangeDet')
    sys.path.insert(0, cd_dir)
    import polygons_change_analyze

    # check projection of the shape file, should be the same
    new_shp_proj4 = map_projection.get_raster_or_vector_srs_info_proj4(shp_list[0])
    for idx in range(len(shp_list)-1):
        shp_proj4 = map_projection.get_raster_or_vector_srs_info_proj4(shp_list[ idx+1 ])
        if shp_proj4 != new_shp_proj4:
            raise ValueError('error, projection insistence between %s and %s'%(new_shp_proj4, shp_proj4))

    polygons_change_analyze.cal_multi_temporal_iou_and_occurrence(shp_list, '')

def postProcess(para_file,inf_post_note, b_skip_getshp=False,test_id=None):
    # test_id is the related to training

    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    # the test string in 'exe.sh'
    test_note = inf_post_note

    WORK_DIR = os.getcwd()

    SECONDS = time.time()

    expr_name = parameters.get_string_parameters(para_file, 'expr_name')
    network_setting_ini = parameters.get_string_parameters(para_file,'network_setting_ini')
    gan_setting_ini = parameters.get_string_parameters_None_if_absence (para_file,'regions_n_setting_image_translation_ini')


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
        map_raster_list_2d = [None] * len(multi_observations)
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
                map_raster_list = []
                for img_idx, img_path in enumerate(inf_img_list):
                    out_shp, out_raster = inf_results_to_shapefile(WORK_DIR, img_idx, area_save_dir, test_id)
                    # if no tif in the output, try to find gpkg
                    if out_shp is None:
                        out_shp = inf_results_gpkg_to_shapefile(WORK_DIR, img_idx, area_save_dir, test_id)
                    if out_shp is None:
                        continue
                    result_shp_list.append(os.path.join(WORK_DIR,out_shp))
                    map_raster_list.append(out_raster)
                if len(result_shp_list) < 1:
                    basic.outputlogMessage(f'Warning, No results for {area_ini}')
                    continue
                # merge shapefiles
                if merge_shape_files(result_shp_list,merged_shp) is False:
                    continue
                map_raster_list_2d[area_idx] = map_raster_list

            merged_shp_list.append(merged_shp)

        if b_skip_getshp is False:
            # add occurrence to each polygons
            get_occurence_for_multi_observation(merged_shp_list)

        for area_idx, area_ini in enumerate(multi_observations):
            area_remark = parameters.get_string_parameters(area_ini, 'area_remark')
            area_save_dir, shp_pre, area_remark_time  = get_observation_save_dir_shp_pre(inf_dir, area_name, area_time, area_remark,test_id)

            merged_shp = os.path.join(WORK_DIR, area_save_dir, shp_pre + '.shp')
            if os.path.isfile(merged_shp) is False:
                print('Warning, %s not exist, skip'%merged_shp)
                continue

            # add attributes to shapefile
            # add_attributes_script = os.path.join(code_dir,'datasets', 'get_polygon_attributes.py')
            shp_attributes = os.path.join(WORK_DIR,area_save_dir, shp_pre+'_post_NOrm.shp')
            # add_polygon_attributes(add_attributes_script,merged_shp, shp_attributes, para_file, area_ini )
            add_polygon_attributes(merged_shp, shp_attributes, para_file, area_ini)

            # remove polygons
            # rm_polygon_script = os.path.join(code_dir,'datasets', 'remove_mappedPolygons.py')
            shp_post = os.path.join(WORK_DIR, area_save_dir, shp_pre+'_post.shp')
            # remove_polygons(rm_polygon_script,shp_attributes, shp_post, para_file)
            remove_polygons_main(shp_attributes, shp_post, para_file)

            # evaluate the mapping results
            # eval_shp_script = os.path.join(code_dir,'datasets', 'evaluation_result.py')
            out_report = os.path.join(WORK_DIR, area_save_dir, shp_pre+'_evaluation_report.txt')
            # evaluation_polygons(eval_shp_script, shp_post, para_file, area_ini,out_report)
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
            if os.path.isfile(out_report):
                io_function.copy_file_to_dst(out_report, bak_eva_report, overwrite=True)
            io_function.copy_file_to_dst(area_ini, bak_area_ini, overwrite=True)

            # copy map raster
            b_backup_map_raster = parameters.get_bool_parameters_None_if_absence(area_ini,'b_backup_map_raster')
            if b_backup_map_raster is True:
                if map_raster_list_2d[area_idx] is not None:
                    for map_tif in map_raster_list_2d[area_idx]:
                        bak_map_tif = os.path.join(backup_dir_area,os.path.basename(map_tif))
                        io_function.copy_file_to_dst(map_tif,bak_map_tif,overwrite=True)

            region_eva_reports[shp_pre] = bak_eva_report



    if len(test_note) > 0:
        bak_para_ini = os.path.join(backup_dir, '_'.join([test_id,'para',test_note]) + '.ini' )
        bak_network_ini = os.path.join(backup_dir, '_'.join([test_id,'network',test_note]) + '.ini' )
        bak_gan_ini = os.path.join(backup_dir, '_'.join([test_id,'gan',test_note]) + '.ini' )
        bak_time_cost = os.path.join(backup_dir, '_'.join([test_id,'time_cost',test_note]) + '.txt' )
    else:
        bak_para_ini = os.path.join(backup_dir, '_'.join([test_id, 'para']) + '.ini')
        bak_network_ini = os.path.join(backup_dir, '_'.join([test_id, 'network']) + '.ini')
        bak_gan_ini = os.path.join(backup_dir, '_'.join([test_id, 'gan']) + '.ini')
        bak_time_cost = os.path.join(backup_dir, '_'.join([test_id, 'time_cost']) + '.txt')
    io_function.copy_file_to_dst(para_file, bak_para_ini,overwrite=True)
    io_function.copy_file_to_dst(network_setting_ini, bak_network_ini,overwrite=True)
    if gan_setting_ini is not None:
        io_function.copy_file_to_dst(gan_setting_ini, bak_gan_ini,overwrite=True)
    if os.path.isfile('time_cost.txt'):
        io_function.copy_file_to_dst('time_cost.txt', bak_time_cost,overwrite=True)

    # output the evaluation report to screen
    for key in region_eva_reports.keys():
        report = region_eva_reports[key]
        if os.path.isfile(report) is False:
            continue
        print('evaluation report for %s:'%key)
        os.system('head -n 7 %s'%report)

    # output evaluation report to table
    if len(test_note) > 0:
        out_table = os.path.join(backup_dir, '_'.join([test_id,'accuracy_table',test_note]) + '.xlsx' )
    else:
        out_table = os.path.join(backup_dir, '_'.join([test_id, 'accuracy_table']) + '.xlsx')
    eva_reports = [ region_eva_reports[key] for key in region_eva_reports if os.path.isfile(region_eva_reports[key])]
    eva_report_to_tables.eva_reports_to_table(eva_reports, out_table)

    duration= time.time() - SECONDS
    os.system('echo "$(date): time cost of post-procesing: %.2f seconds">>time_cost.txt'%duration)


def main(options, args):

    # print("%s : Post-processing" % os.path.basename(sys.argv[0]))

    para_file = args[0]
    # the test string in 'exe.sh'
    if len(args) > 1:
        test_note = args[1]
    else:
        test_note = ''

    postProcess(para_file,test_note)


if __name__ == '__main__':

    usage = "usage: %prog [options] para_file test_note"
    parser = OptionParser(usage=usage, version="1.0 2021-01-22")
    parser.description = 'Introduction: Post-processing  '

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)





