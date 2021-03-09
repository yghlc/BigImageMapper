#!/usr/bin/env python
# Filename: remove_falsePositives 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 11 October, 2019
"""

import os,sys
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(sys.argv[0]), '..')
sys.path.insert(0, code_dir)

import parameters

import basic_src.io_function as io_function
import basic_src.basic as basic
import basic_src.map_projection as map_projection
import vector_features
from vector_features import shape_opeation

def remove_polygons(shapefile,field_name, threshold, bsmaller,output):
    '''
    remove polygons based on attribute values.
    :param shapefile: input shapefile name
    :param field_name:
    :param threshold:
    :param bsmaller:
    :param output:
    :return:
    '''
    operation_obj = shape_opeation()
    if operation_obj.remove_shape_baseon_field_value(shapefile, output, field_name, threshold, smaller=bsmaller) is False:
        return False

def remove_polygons_outside_extent(input_shp, extent_shp, output):
    '''
    remove polygons not in the extent
    :param input_shp:
    :param extent_shp:
    :param output:
    :return:
    '''

    # check projection, must be the same
    input_proj4 = map_projection.get_raster_or_vector_srs_info_proj4(input_shp)
    extent_proj4 = map_projection.get_raster_or_vector_srs_info_proj4(extent_shp)
    if input_proj4 != extent_proj4:
        raise ValueError('error, projection insistence between %s and %s'%(input_shp, extent_shp))

    ## -progress: Only works if input layers have the "fast feature count" capability.
    # ogr2ogr - progress - clipsrc ${extent_shp} ${save_shp} ${input_shp}
    arg_list = ['ogr2ogr', '-progress', '-clipsrc', extent_shp, output, input_shp]
    return basic.exec_command_args_list_one_file(arg_list, output)

def copy_shape_file(input, output):

    return io_function.copy_shape_file(input, output)

def remove_polygons_main(polygons_shp, output, para_file):

    polygons_shp_backup = polygons_shp
    assert io_function.is_file_exist(polygons_shp)

    # remove polygons based on area
    # area_thr = 1000  #10 pixels
    area_thr = parameters.get_digit_parameters_None_if_absence(para_file,'minimum_area','int')
    b_smaller = True
    if area_thr is not None:
        rm_area_save_shp = io_function.get_name_by_adding_tail(polygons_shp_backup, 'rmArea')
        if remove_polygons(polygons_shp, 'INarea', area_thr, b_smaller, rm_area_save_shp) is False:
            basic.outputlogMessage("error, removing polygons based on size failed")
        else:
            polygons_shp = rm_area_save_shp
    else:
        basic.outputlogMessage('warning, minimum_area is absent in the para file, skip removing polygons based on areas')

    # remove  polygons based on slope information
    # slope_small_thr = 2
    slope_small_thr = parameters.get_digit_parameters_None_if_absence(para_file,'minimum_slope','float')
    b_smaller = True
    if slope_small_thr is not None:
        rm_slope_save_shp1 = io_function.get_name_by_adding_tail(polygons_shp_backup, 'rmslope1')
        if remove_polygons(polygons_shp, 'slo_mean', slope_small_thr, b_smaller, rm_slope_save_shp1) is False:
            basic.outputlogMessage("error, removing polygons based on slo_mean failed")
        else:
            polygons_shp = rm_slope_save_shp1
    else:
        basic.outputlogMessage('warning, minimum_slope is absent in the para file, skip removing polygons based on minimum slope')

    # slope_large_thr = 20
    slope_large_thr = parameters.get_digit_parameters_None_if_absence(para_file,'maximum_slope','float')
    b_smaller = False
    if slope_large_thr is not None:
        rm_slope_save_shp2 = io_function.get_name_by_adding_tail(polygons_shp_backup, 'rmslope2')
        if remove_polygons(polygons_shp, 'slo_mean', slope_large_thr, b_smaller, rm_slope_save_shp2) is False:
            basic.outputlogMessage("error, removing polygons based on slo_mean (2) failed")
        else:
            polygons_shp = rm_slope_save_shp2
    else:
        basic.outputlogMessage('warning, maximum_slope is absent in the para file, skip removing polygons based on maximum slope')

    # remove polygons based on dem
    # dem_small_thr = 3000
    dem_small_thr = parameters.get_digit_parameters_None_if_absence(para_file,'minimum_elevation','int')
    b_smaller = True
    if dem_small_thr is not None:
        rm_dem_save_shp = io_function.get_name_by_adding_tail(polygons_shp_backup, 'rmDEM')
        if remove_polygons(polygons_shp, 'dem_mean', dem_small_thr, b_smaller, rm_dem_save_shp) is False:
            basic.outputlogMessage("error, removing polygons based on dem_mean failed")
        else:
            polygons_shp = rm_dem_save_shp
    else:
        basic.outputlogMessage('warning, minimum_elevation is absent in the para file, skip removing polygons based on minimum elevation')

    # remove polygons based elevation reduction
    minimum_dem_reduction_area_thr = parameters.get_digit_parameters_None_if_absence(para_file,'minimum_dem_reduction_area','float')
    b_smaller = True
    if minimum_dem_reduction_area_thr is not None:
        rm_demD_save_shp = io_function.get_name_by_adding_tail(polygons_shp_backup, 'rmdemD')
        if remove_polygons(polygons_shp, 'demD_area', minimum_dem_reduction_area_thr, b_smaller, rm_demD_save_shp) is False:
            basic.outputlogMessage("error, removing polygons based on demD_area failed")
        else:
            polygons_shp = rm_demD_save_shp
    else:
        basic.outputlogMessage('warning, minimum_dem_reduction_area is absent in the para file, skip removing polygons based on minimum_dem_reduction_area')


    # remove polygons based on occurrence
    min_ocurr = parameters.get_digit_parameters_None_if_absence(para_file,'threshold_occurrence_multi_observation','int')
    b_smaller = True
    if min_ocurr is not None:
        rm_occur_save_shp = io_function.get_name_by_adding_tail(polygons_shp_backup,'RmOccur')
        if remove_polygons(polygons_shp, 'time_occur', min_ocurr, b_smaller, rm_occur_save_shp) is False:
            basic.outputlogMessage("error, removing polygons based on time_occur failed")
        else:
            polygons_shp = rm_occur_save_shp
    else:
        basic.outputlogMessage('warning, threshold_occurrence_multi_observation is absent in the para file, '
                               'skip removing polygons based on it')

    # remove polygons not in the extent
    outline_shp = parameters.get_string_parameters_None_if_absence(para_file,'target_outline_shp')
    if outline_shp is not None:
        rm_outline_save_shp = io_function.get_name_by_adding_tail(polygons_shp_backup, 'rmOutline')
        remove_polygons_outside_extent(polygons_shp, outline_shp, rm_outline_save_shp)
        polygons_shp = rm_outline_save_shp
    else:
        basic.outputlogMessage('warning, target_outline_shp is absent in the para file, skip removing polygons based on outlines')


    # copy to final output
    copy_shape_file(polygons_shp,output)

    pass

def main(options, args):
    polygons_shp = args[0]

    output = options.output
    if output is None:
        output = io_function.get_name_by_adding_tail(polygons_shp,'removed')
    para_file = options.para_file

    remove_polygons_main(polygons_shp, output, para_file)




if __name__ == "__main__":
    usage = "usage: %prog [options] shp_file"
    parser = OptionParser(usage=usage, version="1.0 2019-1-4")
    parser.description = 'Introduction: remove polygons based on an attributes values'

    parser.add_option("-o", "--output",
                      action="store", dest="output",#default='save_polygon.shp',
                      help="save file path")

    parser.add_option("-p", "--para_file",
                      action="store", dest="para_file",
                      help="the parameters file")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    if options.para_file is None:
        print('error, parameter file is required')
        sys.exit(2)

    main(options, args)