#!/usr/bin/env python
# Filename: add_info2Pylygons 
"""
introduction: add information to polygons as new attribute

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 25 February, 2019
"""

import os,sys
from optparse import OptionParser

HOME = os.path.expanduser('~')
# path of DeeplabforRS
codes_dir2 = HOME + '/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)

import basic_src.io_function as io_function
import basic_src.basic as basic
import vector_features
from vector_features import shape_opeation

def add_raster_info_from_bufferArea(polygons_shp,raster_file,raster_name, b_buffer_size):
    """
    calculate the raster information such elevation, then add toeach polygon
    Args:
        polygons_shp: input shapfe file
        raster_file:  raster file, should have the same projection of shapefile
        raster_name: the name of raster, should less than four letters, will be used as part of the attribute name
        b_buffer_size: the size of buffer area in meters

    Returns: True if successful, False Otherwise
    """
    if io_function.is_file_exist(polygons_shp) is False:
        return False
    if io_function.is_file_exist(raster_file) is False:
        return False
    operation_obj = shape_opeation()

    ## calculate the topography information from the buffer area
    basic.outputlogMessage("info: calculate the raster information from the buffer area")
    buffer_polygon_shp = io_function.get_name_by_adding_tail(polygons_shp, 'buffer')
    # if os.path.isfile(buffer_polygon_shp) is False:
    if vector_features.get_buffer_polygons(polygons_shp,buffer_polygon_shp,b_buffer_size) is False:
        raise IOError("error, failed in producing the buffer_polygon_shp")

    # replace the polygon shape file
    polygons_shp_backup = polygons_shp
    polygons_shp = buffer_polygon_shp

    # all_touched: bool, optional
    #     Whether to include every raster cell touched by a geometry, or only
    #     those having a center point within the polygon.
    #     defaults to `False`
    #   Since the dem usually is coarser, so we set all_touched = True
    all_touched = True

    stats_list = ['min', 'max', 'mean', 'std'] #['min', 'max', 'mean', 'count','median','std']
    if operation_obj.add_fields_from_raster(polygons_shp, raster_file, raster_name, band=1,
                                            stats_list=stats_list,all_touched=all_touched) is False:
        return False

    # copy the information to the original shape file
    operation_obj.add_fields_shape(polygons_shp_backup, buffer_polygon_shp, polygons_shp_backup)

    return True


def add_raster_info_insidePolygons(polygons_shp,raster_file,raster_name):
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
    if operation_obj.add_fields_from_raster(polygons_shp, raster_file, raster_name, band=1, stats_list=stats_list,
                                                all_touched=all_touched) is False:
        return False

    return True

def check_same_projection(shp_file, raster_file):
    '''
    check the projection of shape file and the raster file
    :param shp_file:
    :param raster_file:
    :return:
    '''

    shp_args_list = ['gdalsrsinfo','-o','epsg',shp_file]
    shp_epsg_str = basic.exec_command_args_list_one_string(shp_args_list)

    raster_args_list = ['gdalsrsinfo','-o','epsg',raster_file]
    raster_epsg_str = basic.exec_command_args_list_one_string(raster_args_list)

    if shp_epsg_str == raster_epsg_str:
        return True
    else:
        return False

def add_IoU_values(polygons_shp,ground_truth_shp,field_name):
    '''
    add IoU values to the shape file
    :param polygons_shp:
    :param ground_truth_shp:
    :param field_name:  should be 'IoU'
    :return:
    '''
    IoUs = vector_features.calculate_IoU_scores(polygons_shp, ground_truth_shp)
    if IoUs is False:
        return False
    # save IoU to result shape file
    operation_obj = shape_opeation()
    return operation_obj.add_one_field_records_to_shapefile(polygons_shp, IoUs, field_name)

def add_adjacent_polygon_count(polygons_shp,buffer_size,field_name):
    '''

    :param polygons_shp:
    :param buffer_size:
    :param field_name: should be "adj_count"
    :return:
    '''
    # save IoU to result shape file
    operation_obj = shape_opeation()
    counts = vector_features.get_adjacent_polygon_count(polygons_shp,buffer_size)
    # print(len(counts))
    return operation_obj.add_one_field_records_to_shapefile(polygons_shp, counts, field_name)



def main(options, args):

    polygons_shp = args[0]
    field_name = options.field_name

    # add information from a raster file
    raster_file = options.raster_file
    if raster_file is not None:
        if check_same_projection(polygons_shp,raster_file) is False:
            raise ValueError('%s and %s don\'t have the same projection')

        buffer_meters = options.buffer_meters
        if buffer_meters is None:
            add_raster_info_insidePolygons(polygons_shp, raster_file, field_name)
            basic.outputlogMessage('add %s information (inside polygons) to %s'%(field_name,polygons_shp))
        else:
            add_raster_info_from_bufferArea(polygons_shp, raster_file, field_name, buffer_meters)
            basic.outputlogMessage('add %s information (in surrounding buffer area) to %s' % (field_name, polygons_shp))

    # add IoU values
    validation_shp = options.val_polygon
    if validation_shp is not None:
        if check_same_projection(polygons_shp,validation_shp) is False:
            raise ValueError('%s and %s don\'t have the same projection')

        if add_IoU_values(polygons_shp,validation_shp,field_name):
            basic.outputlogMessage('add %s to %s' % (field_name, polygons_shp))

    if field_name=="adj_count":
        buffer_meters = options.buffer_meters
        add_adjacent_polygon_count(polygons_shp, buffer_meters, field_name)


if __name__ == "__main__":
    usage = "usage: %prog [options] shp_file"
    parser = OptionParser(usage=usage, version="1.0 2019-1-4")
    parser.description = 'Introduction: add information to polygons'

    parser.add_option("-r", "--raster",
                      action="store", dest="raster_file",
                      help="the path of the raster file")

    parser.add_option("-v", "--val_polygon",
                      action="store", dest="val_polygon",
                      help="the path of validation polygons, for calculating IoU values")

    parser.add_option("-n", "--field_name",
                      action="store", dest="field_name",
                      help="for raster, it should less than four letters, will be used as part of the attribute name")

    parser.add_option("-b", "--buffer",
                      action="store", dest="buffer_meters",type=float,
                      help="the buffer area in meters, if this is assigned, it will calculate the info in the buffer around "
                           "the polygon, otherwise in the polygon")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)