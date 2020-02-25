#!/usr/bin/env python3
# Filename: rm_polygons_multTemporal 
"""
introduction: remove mappping polygons based on multi-temporal results

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 23 February, 2020
"""


import sys,os
from optparse import OptionParser

# added path of DeeplabforRS
deeplabRS = os.path.expanduser('~/codes/PycharmProjects/DeeplabforRS')
sys.path.insert(0, deeplabRS)
import basic_src.basic as basic
import basic_src.map_projection as map_projection
import basic_src.io_function as io_function
import parameters

# add ChangeDet_DL folder
sys.path.insert(0, os.path.expanduser('~/codes/PycharmProjects/ChangeDet_DL/thawSlumpChangeDet'))
import remove_nonActive_thawSlumps
import polygons_change_analyze


import re

def main(options, args):

    # get multi-temporal shapefile list
    para_file = options.para_file
    b_remove = parameters.get_bool_parameters_None_if_absence(para_file,'b_remove_polygons_using_multitemporal_results')
    # exit
    if b_remove is None or b_remove is False:
        basic.outputlogMessage('Warning, b_remove_polygons_using_multitemporal_results not set or is NO')
        return True

    shp_dir = args[0]
    file_pattern = args[1]
    polyon_shps_list = io_function.get_file_list_by_pattern(shp_dir,file_pattern)
    if len(polyon_shps_list) < 2:
        raise ValueError('Error, less than two shapefiles, cannot conduct multi-polygon anlysis')

    # make polyon_shps_list in order: I0 to In
    polyon_shps_list.sort(key=lambda x: int(re.findall('I\d+',os.path.basename(x))[0][1:]))

    # print(polyon_shps_list)
    # sys.exit()

    # check projection of the shape file, should be the same
    new_shp_proj4 = map_projection.get_raster_or_vector_srs_info_proj4(polyon_shps_list[0])
    for idx in range(len(polyon_shps_list)-1):
        shp_proj4 = map_projection.get_raster_or_vector_srs_info_proj4(polyon_shps_list[ idx+1 ])
        if shp_proj4 != new_shp_proj4:
            raise ValueError('error, projection insistence between %s and %s'%(new_shp_proj4, shp_proj4))

    # polygon change analysis
    polygons_change_analyze.cal_multi_temporal_iou_and_occurrence(polyon_shps_list, para_file)

    # remove non active polygons
    remove_nonActive_thawSlumps.remove_non_active_thaw_slumps(polyon_shps_list, para_file)

    # back up files and conduct evaluation
    for idx, shp_path in enumerate(polyon_shps_list):

        # evaluation files
        shp_rmTimeiou = io_function.get_name_by_adding_tail(shp_path,'rmTimeiou')
        basic.outputlogMessage('(%d/%d) evaluation of %s'%(idx,len(polyon_shps_list),shp_rmTimeiou))

        # evaluation
        args_list = [os.path.join(deeplabRS,'evaluation_result.py'), '-p', para_file, shp_rmTimeiou]
        if basic.exec_command_args_list_one_file(args_list,'evaluation_report.txt') is False:
            return False

        I_idx_str = re.findall('I\d+', os.path.basename(shp_rmTimeiou))

        old_eva_report = io_function.get_file_list_by_pattern(shp_dir, I_idx_str[0]+'*eva_report*'+'.txt')
        old_eva_report = [ item for item in old_eva_report if 'rmTimeiou' not in item]

        old_eva_report_name = old_eva_report[0]

        eva_report_name = io_function.get_name_by_adding_tail(old_eva_report_name,'rmTimeiou')
        # io_function.move_file_to_dst(old_eva_report,backup_eva_report)
        # io_function.move_file_to_dst('evaluation_report.txt', old_eva_report)
        io_function.move_file_to_dst('evaluation_report.txt', eva_report_name,overwrite=True)

        # back up the shape files (no_need)

    basic.outputlogMessage('Finish removing polygons using multi-temporal mapping results')



if __name__ == "__main__":
    usage = "usage: %prog [options] shape_file_dir file_pattern "
    parser = OptionParser(usage=usage, version="1.0 2020-02-23")
    parser.description = 'Introduction: remove polygons (e.g., non-active polygons) based on multi-temporal polygons '

    parser.add_option("-p", "--para",
                      action="store", dest="para_file",
                      help="the parameters file")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    # set parameters files
    if options.para_file is None:
        print('error, no parameters file')
        parser.print_help()
        sys.exit(2)
    else:
        parameters.set_saved_parafile_path(options.para_file)

    basic.setlogfile('polygons_remove_nonActiveRTS.log')

    main(options, args)