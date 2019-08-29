#!/usr/bin/env python
# Filename: add_info2Pylygons 
"""
introduction: keep the true positive only, i.e., remove polygons with IOU less than or equal to 0.5.

it can also be used to remove other polygons based on an attribute

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 26 February, 2019
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


def check_same_projection(shp_file, file2):
    '''
    check the projection of shape file and the raster file
    :param shp_file:
    :param raster_file:
    :return:
    '''

    shp_args_list = ['gdalsrsinfo','-o','epsg',shp_file]
    shp_epsg_str = basic.exec_command_args_list_one_string(shp_args_list)

    raster_args_list = ['gdalsrsinfo','-o','epsg',file2]
    raster_epsg_str = basic.exec_command_args_list_one_string(raster_args_list)

    if shp_epsg_str == raster_epsg_str:
        return True
    else:
        return False

def remove_polygons(shapefile,field_name, threshold, bsmaller,output):
    #  remove the not narrow polygon based on ratio_p_a
    operation_obj = shape_opeation()
    if operation_obj.remove_shape_baseon_field_value(shapefile, output, field_name, threshold, smaller=bsmaller) is False:
        return False

def remove_lines_based_on_polygons(shp_line,output_mainline,shp_polygon):
    '''
    if lines if they don't overlap any polygons
    :param shp_line:
    :param output_mainline:
    :param shp_polygon:
    :return:
    '''
    if check_same_projection(shp_line,shp_polygon) is False:
        raise ValueError('%s and %s don\'t have the same projection')

    print(shp_line,shp_polygon)
    inte_lines_list = vector_features.get_intersection_of_line_polygon(shp_line,shp_polygon)
    b_remove = [True if item.is_empty else False for item in inte_lines_list ]
    # print(b_remove)
    # a=0
    # b=0
    # for rm in b_remove:
    #     if rm is True:
    #         a += 1
    #     else:
    #         b += 1
    # print(a,b)

    #note that, after remove, the number of main lines are greater than the number of polygons in "shp_polygon"
    #This is because, in Beiluhe, some mapped thaw slumps close to each other were merged to one
    operation_obj = shape_opeation()
    if operation_obj.remove_shapes_by_list(shp_line,output_mainline,b_remove) is False:
        return False


def remove_polygons_intersect_multi_ground_truths(shp_file, shp_ground_truth, output, copy_fields=None):
    '''

    :param shp_file:
    :param shp_ground_truth:
    :param output:
    :param copy_fields:
    :return:
    '''
    operation_obj = shape_opeation()
    return operation_obj.remove_polygons_intersect_multi_polygons(shp_file, shp_ground_truth, output, copy_fields=copy_fields)



def main(options, args):
    polygons_shp = args[0]

    field_name = options.field_name
    threshold = options.threshold
    bsmaller = options.bsmaller  # if true, then remove the bsmaller ones
    output = options.output

    # print(field_name,threshold,bsmaller,output)
    if field_name is not None:
        remove_polygons(polygons_shp, field_name, threshold, bsmaller, output)


    # remove the file in main_lines
    shp_mainline = options.shp_mainline
    if shp_mainline is not None:
        output_mainline = options.output_mainline
        remove_lines_based_on_polygons(shp_mainline, output_mainline, output)

    # remove polygon based on the intersection with ground truth polygons
    val_polygon = options.val_polygon
    if val_polygon is not None:
        copy_fields = options.copy_fields
        copy_fields = copy_fields.split(',')
        remove_polygons_intersect_multi_ground_truths(polygons_shp, val_polygon, output, copy_fields=copy_fields)

if __name__ == "__main__":
    usage = "usage: %prog [options] shp_file"
    parser = OptionParser(usage=usage, version="1.0 2019-1-4")
    parser.description = 'Introduction: remove polygons based on an attributes values'

    parser.add_option("-o", "--output",
                      action="store", dest="output",default='save_polygon.shp',
                      help="save file path")

    parser.add_option("-f", "--field_name",
                      action="store", dest="field_name",
                      help="the field name of the attribute based on which to remove polygons")

    parser.add_option("-t", "--threshold",
                      action="store", dest="threshold", default=0.5,type=float,
                      help="the threshold to remove polygons")

    parser.add_option("-l", "--mainline",
                      action="store", dest="shp_mainline",
                      help="the shape file store the main_Line of polygons")

    parser.add_option("-m", "--output_mainline",
                      action="store", dest="output_mainline",default='save_mainline.shp',
                      help="save file path of main line")

    parser.add_option("-v", "--val_polygon",
                      action="store", dest="val_polygon",
                      help="the path of validation polygons")

    parser.add_option("-c", "--copy_fields",
                      action="store", dest="copy_fields",
                      help="the multi field names to be copied from validation polygons, e.g., 'area,perimeter', use comma to sperate them but no space")

    parser.add_option("-s", "--bsmaller",
                      action="store_true", dest="bsmaller",
                      help="True will remove the bsmaller ones")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)