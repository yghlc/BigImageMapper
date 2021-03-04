#!/usr/bin/env python
# Filename: post_process 
"""
introduction:  calculate polygon attributes, including size, perimeter, also mean slope & dem  etc.

copy and modified from polygon_post_process.py

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 27 March, 2017
modified: 23 January, 2021
"""

import os, sys
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(sys.argv[0]), '..')
sys.path.insert(0, code_dir)
import parameters
import basic_src.basic as basic
import basic_src.io_function as io_function

import math

# import  parameters
import vector_features
import vector_gpd
from vector_features import shape_opeation
import parameters
from raster_statistic import zonal_stats_multiRasters


def remove_nonclass_polygon(input_shp,output_shp, field_name='svmclass'):
    """
    remove polygon which is not belong to target
    :param input_shp: input shape file
    :param output_shp: output shape file
    :param field_name: the field name of specific field containing class information in the shape file
    :return: True if successful, False Otherwise
    """
    operation_obj = shape_opeation()
    if operation_obj.remove_nonclass_polygon(input_shp, output_shp, field_name):
        operation_obj = None
        return True
    else:
        operation_obj = None
        return False

def merge_polygons_in_gully(input_shp, output_shp):
    """
    merge polygons in one gully. supposed that the polygons touch each other are belong to the same gully
    :param input_shp: input shapfe file
    :param output_shp: output shape file contains the merged polygons
    :return: True if successful, False Otherwise
    """
    return vector_features.merge_touched_polygons_in_shapefile(input_shp,output_shp )

def cal_add_area_length_of_polygon(input_shp):
    """
    calculate the area, perimeter of polygons, save to the original file
    :param input_shp: input shapfe file
    :return: True if successful, False Otherwise
    """
    return vector_features.cal_area_length_of_polygon(input_shp )

def calculate_polygon_topography(polygons_shp,para_file, dem_files,slope_files,aspect_files=None, dem_diffs=None):
    """
    calculate the topography information such elevation and slope of each polygon
    Args:
        polygons_shp: input shapfe file
        dem_files: DEM raster file or tiles, should have the same projection of shapefile
        slope_files: slope raster file or tiles  (can be drived from dem file by using QGIS or ArcGIS)
        aspect_files: aspect raster file or tiles (can be drived from dem file by using QGIS or ArcGIS)

    Returns: True if successful, False Otherwise
    """
    if io_function.is_file_exist(polygons_shp) is False:
        return False
    operation_obj = shape_opeation()

    ## calculate the topography information from the buffer area

    # the para file was set in parameters.set_saved_parafile_path(options.para_file)
    b_use_buffer_area = parameters.get_bool_parameters(para_file,'b_topo_use_buffer_area')

    if b_use_buffer_area is True:

        b_buffer_size = 5  # meters (the same as the shape file)

        basic.outputlogMessage("info: calculate the topography information from the buffer area")
        buffer_polygon_shp = io_function.get_name_by_adding_tail(polygons_shp, 'buffer')
        # if os.path.isfile(buffer_polygon_shp) is False:
        if vector_features.get_buffer_polygons(polygons_shp,buffer_polygon_shp,b_buffer_size) is False:
            basic.outputlogMessage("error, failed in producing the buffer_polygon_shp")
            return False
        # else:
        #     basic.outputlogMessage("warning, buffer_polygon_shp already exist, skip producing it")
        # replace the polygon shape file
        polygons_shp_backup = polygons_shp
        polygons_shp = buffer_polygon_shp
    else:
        basic.outputlogMessage("info: calculate the topography information from the inside of each polygon")



    # all_touched: bool, optional
    #     Whether to include every raster cell touched by a geometry, or only
    #     those having a center point within the polygon.
    #     defaults to `False`
    #   Since the dem usually is coarser, so we set all_touched = True
    all_touched = True
    process_num = 4

    # #DEM
    if dem_files is not None:
        stats_list = ['min', 'max','mean','median','std']            #['min', 'max', 'mean', 'count','median','std']
        # if operation_obj.add_fields_from_raster(polygons_shp, dem_file, "dem", band=1,stats_list=stats_list,all_touched=all_touched) is False:
        #     return False
        if zonal_stats_multiRasters(polygons_shp,dem_files,stats=stats_list,prefix='dem',band=1,all_touched=all_touched, process_num=process_num) is False:
            return False
    else:
        basic.outputlogMessage("warning, DEM file not exist, skip the calculation of DEM information")

    # #slope
    if slope_files is not None:
        stats_list = ['min', 'max','mean', 'median', 'std']
        if zonal_stats_multiRasters(polygons_shp,slope_files,stats=stats_list,prefix='slo',band=1,all_touched=all_touched, process_num=process_num) is False:
            return False
    else:
        basic.outputlogMessage("warning, slope file not exist, skip the calculation of slope information")

    # #aspect
    if aspect_files is not None:
        stats_list = ['min', 'max','mean', 'std']
        if zonal_stats_multiRasters(polygons_shp,aspect_files,stats=stats_list,prefix='asp',band=1,all_touched=all_touched, process_num=process_num) is False:
            return False
    else:
        basic.outputlogMessage('warning, aspect file not exist, ignore adding aspect information')

    # elevation difference
    if dem_diffs is not None:
        stats_list = ['min', 'max', 'mean', 'median', 'std','area']
        # only count the pixel within this range when do statistics
        dem_diff_range_str = parameters.get_string_list_parameters(para_file, 'dem_difference_range')
        range = [ None if item.upper() == 'NONE' else float(item) for item in dem_diff_range_str ]

        # expand the polygon when doing dem difference statistics
        buffer_size_dem_diff = parameters.get_digit_parameters(para_file, 'buffer_size_dem_diff','float')

        if zonal_stats_multiRasters(polygons_shp,dem_diffs,stats=stats_list,prefix='demD',band=1,all_touched=all_touched, process_num=process_num,
                                    range=range, buffer=buffer_size_dem_diff) is False:
            return False
    else:
        basic.outputlogMessage('warning, dem difference file not exist, ignore adding dem diff information')

    # # hillshape

    # copy the topography information
    if b_use_buffer_area is True:
        operation_obj.add_fields_shape(polygons_shp_backup, buffer_polygon_shp, polygons_shp_backup)

    return True

def calculate_hydrology(polygons_shp,flow_accumulation):
    """
    calculate the hydrology information of each polygons
    Args:
        polygons_shp:  input shapfe file
        flow_accumulation: the file path of flow accumulation

    Returns: True if successful, False Otherwise

    """
    if io_function.is_file_exist(polygons_shp) is False:
        return False
    operation_obj = shape_opeation()

    # all_touched: bool, optional
    #     Whether to include every raster cell touched by a geometry, or only
    #     those having a center point within the polygon.
    #     defaults to `False`
    #   Since the dem usually is coarser, so we set all_touched = True
    all_touched = True

    # #DEM

    stats_list = ['min', 'max', 'mean', 'std']  # ['min', 'max', 'mean', 'count','median','std']
    if operation_obj.add_fields_from_raster(polygons_shp, flow_accumulation, "F_acc", band=1, stats_list=stats_list,
                                                all_touched=all_touched) is False:
        return False


    pass

def calculate_gully_information(gullies_shp):
    """
    get Oriented minimum bounding box for the gully polygon shapefile,
    and update the shape information based on oriented minimum bounding box to
        the gullies_shp
    :param gullies_shp: input shapefile contains the gully polygons
    :return: True if successful, False Otherwise
    """
    operation_obj = shape_opeation()
    output_shapeinfo = io_function.get_name_by_adding_tail(gullies_shp, 'shapeInfo')
    if os.path.isfile(output_shapeinfo) is False:
        operation_obj.get_polygon_shape_info(gullies_shp, output_shapeinfo)
        # note: the area in here, is the area of the oriented minimum bounding box, not the area of polygon
        operation_obj.add_fields_shape(gullies_shp, output_shapeinfo, gullies_shp)
    else:
        basic.outputlogMessage('warning, %s already exist, skip calculate and add shape feature' % output_shapeinfo)
    # put all feature to one shapefile
    # parameter 3 the same as parameter 1 to overwrite the input file

    # add width/height (suppose height greater than width)
    width_height_list = operation_obj.get_shape_records_value(gullies_shp,attributes=['WIDTH','HEIGHT'])
    if width_height_list is not False:
        ratio = []
        for width_height in width_height_list:
            if width_height[0] > width_height[1]:
                r_value = width_height[1] / width_height[0]
            else:
                r_value = width_height[0] / width_height[1]
            ratio.append(r_value)
        operation_obj.add_one_field_records_to_shapefile(gullies_shp,ratio,'ratio_w_h')

    # add perimeter/area
    perimeter_area_list = operation_obj.get_shape_records_value(gullies_shp, attributes=['INperimete','INarea'])
    if perimeter_area_list is not False:
        ratio_p_a = []
        for perimeter_area in perimeter_area_list:
            try:
                r_value = (perimeter_area[0])**2 / perimeter_area[1]
            except ZeroDivisionError:
                basic.outputlogMessage('warning, ZeroDivisionError: float division by zero')
                r_value = 0
            ratio_p_a.append(r_value)
        operation_obj.add_one_field_records_to_shapefile(gullies_shp, ratio_p_a, 'ratio_p_a')

    # add circularity (4*pi*area/perimeter**2) which is similar to ratio_p_a
    circularity = []
    for perimeter_area in perimeter_area_list:
        value = (4*math.pi*perimeter_area[1] / perimeter_area[0] ** 2)
        circularity.append(value)
    operation_obj.add_one_field_records_to_shapefile(gullies_shp, circularity, 'circularit')

    return True

def remove_small_round_polygons(input_shp,output_shp,area_thr,ratio_thr):
    """
    remove the polygons that is not gully, that is the polygon is too small or not narrow.
    # too small or not narrow
    :param input_shp: input shape file
    :param output_shp:  output  shape file
    :return: True if successful, False otherwise
    """

    #remove the too small polygon
    operation_obj = shape_opeation()
    output_rm_small = io_function.get_name_by_adding_tail(input_shp,'rmSmall')
    # area_thr = parameters.get_minimum_gully_area()
    if operation_obj.remove_shape_baseon_field_value(input_shp,output_rm_small,'INarea',area_thr,smaller=True) is False:
        return False

    # remove the not narrow polygon
    # it seems that this can not represent how narrow the polygon is, because they are irregular polygons
    # whatever, it can remove some flat, and not long polygons. if you want to omit this, just set the maximum_ratio_width_height = 1

    output_rm_Rwh=io_function.get_name_by_adding_tail(input_shp,'rmRwh')
    ratio_thr = parameters.get_maximum_ratio_width_height()
    if operation_obj.remove_shape_baseon_field_value(output_rm_small, output_rm_Rwh, 'ratio_w_h', ratio_thr, smaller=False) is False:
        return False

    #  remove the not narrow polygon based on ratio_p_a
    ratio_thr = parameters.get_minimum_ratio_perimeter_area()
    if operation_obj.remove_shape_baseon_field_value(output_rm_Rwh, output_shp, 'ratio_p_a', ratio_thr, smaller=True) is False:
        return False

    return True

def get_file_path_parameter(parafile, data_dir, data_name_or_pattern):

    data_dir = parameters.get_directory_None_if_absence(parafile, data_dir)
    data_name_or_pattern = parameters.get_string_parameters_None_if_absence(parafile, data_name_or_pattern)
    if data_dir is None or data_name_or_pattern is None:
        return None
    file_list = io_function.get_file_list_by_pattern(data_dir,data_name_or_pattern)

    if len(file_list) < 1:
        raise IOError('NO file in %s with name or pattern: %s'%(data_dir, data_name_or_pattern))
    if len(file_list) == 1:
        return file_list[0]
    else:
        # return multiple files
        return file_list


def get_topographic_files(data_para_file):

    dem_files = get_file_path_parameter(data_para_file,'dem_file_dir', 'dem_file_or_pattern')
    slope_files = get_file_path_parameter(data_para_file,'slope_file_dir', 'slope_file_or_pattern')
    aspect_files = get_file_path_parameter(data_para_file, 'aspect_file_dir', 'aspect_file_or_pattern')
    dem_diff_files = get_file_path_parameter(data_para_file, 'dem_diff_file_dir', 'dem_diff_file_or_pattern')

    return dem_files, slope_files, aspect_files,dem_diff_files

def add_polygon_attributes(input, output, para_file, data_para_file):

    if io_function.is_file_exist(input) is False:
        return False

    # copy output
    if io_function.copy_shape_file(input, output) is False:
        raise IOError('copy shape file %s failed'%input)

    # remove narrow parts of mapped polygons
    polygon_narrow_part_thr = parameters.get_digit_parameters_None_if_absence(para_file, 'mapped_polygon_narrow_threshold', 'float')
    #  if it is not None, then it will try to remove narrow parts of polygons
    if polygon_narrow_part_thr is not None and polygon_narrow_part_thr > 0:
        # use the buffer operation to remove narrow parts of polygons
        basic.outputlogMessage("start removing narrow parts (thr %.2f) in polygons"%(polygon_narrow_part_thr*2))
        if vector_gpd.remove_narrow_parts_of_polygons_shp_NOmultiPolygon(input, output, polygon_narrow_part_thr):
            message = "Finished removing narrow parts (thr %.2f) in polygons and save to %s"%(polygon_narrow_part_thr*2,output)
            basic.outputlogMessage(message)
        else:
            pass
    else:
        basic.outputlogMessage("warning, mapped_polygon_narrow_threshold is not in the parameter file, skip removing narrow parts")

    # calculate area, perimeter of polygons
    if cal_add_area_length_of_polygon(output) is False:
        return False

    # calculate the polygon information
    b_calculate_shape_info = parameters.get_bool_parameters_None_if_absence(para_file,'b_calculate_shape_info')
    if b_calculate_shape_info:
        # remove "_shapeInfo.shp" to make it calculate shape information again
        os.system('rm *_shapeInfo.shp')
        if calculate_gully_information(output) is False:
            return False


    # add topography of each polygons
    dem_files, slope_files, aspect_files, dem_diff_files = get_topographic_files(data_para_file)
    if calculate_polygon_topography(output,para_file,dem_files,slope_files,aspect_files=aspect_files,dem_diffs=dem_diff_files) is False:
        basic.outputlogMessage('Warning: calculate information of topography failed')
        # return False   #  don't return


    return True


def main(options, args):
    input = args[0]
    output = args[1]

    data_para_file = options.data_para
    if data_para_file is None:
        data_para_file = options.para_file

    add_polygon_attributes(input, output, options.para_file, data_para_file)


if __name__=='__main__':



    usage = "usage: %prog [options] input_path output_file"
    parser = OptionParser(usage=usage, version="1.0 2017-7-24")
    parser.description = 'Introduction: Post process of Polygon shape file, including  ' \
                         'statistic polygon information,' 
    parser.add_option("-p", "--para",
                      action="store", dest="para_file",
                      help="the parameters file")

    parser.add_option("-d", "--data_para",
                      action="store", dest="data_para",
                      help="the parameters file for data")

    parser.add_option("-a", "--min_area",
                      action="store", dest="min_area",type=float,
                      help="the minimum for each polygon")
    parser.add_option("-r", "--min_ratio",
                      action="store", dest="min_ratio",type=float,
                      help="the minimum ratio (perimeter*perimeter / area) for each polygon (thin and long polygon has larger ratio)")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 2:
        parser.print_help()
        sys.exit(2)
    ## set parameters files
    if options.para_file is None:
        print('error, no parameters file')
        parser.print_help()
        sys.exit(2)
    else:
        parameters.set_saved_parafile_path(options.para_file)

    # test
    # ouput_merged = args[0]
    # dem_file = parameters.get_dem_file()
    # slope_file = parameters.get_slope_file()
    # calculate_polygon_topography(ouput_merged,dem_file,slope_file)



    main(options, args)
