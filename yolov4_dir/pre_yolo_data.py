#!/usr/bin/env python
# Filename: pre_yolo_data 
"""
introduction: conver training data for semantic to yolo objection.

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 04 April, 2021
"""

import os,sys
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import basic_src.io_function as io_function
import parameters
import time

import datasets.raster_io as raster_io
from datasets.get_boxes_label_images import get_boxes_from_label_image
import datasets.vector_gpd as vector_gpd
import pandas as pd
import basic_src.map_projection as map_projection
from yoltv4Based.yolt_func import convert

from yoltv4Based.yolt_func import convert_reverse

def get_yolo_boxes_one_img(idx, total, image_path, label_path,num_classes_noBG,rm_edge_obj=False,b_save_vector=False):
    print('to yolo box: %d/%d'%(idx+1, total))
    # a object : [ class_id,  minX, minY, maxX, maxY ]
    objects = get_boxes_from_label_image(label_path)
    save_object_txt = os.path.splitext(image_path)[0] + '.txt'
    height, width, count, dtype = raster_io.get_height_width_bandnum_dtype(label_path)

    with open(save_object_txt, 'w') as f_obj:
        obj_polygons = []
        for object in objects:
            class_id, minX, minY, maxX, maxY = object
            if class_id > num_classes_noBG:
                raise ValueError('Class ID: %d greater than number of classes in label: %s'%(class_id, label_path))

            # remove objects touch the image edge, usually, target object is in the center of images,
            # but some targets close to this one may be cut when we extract sub-images or split imgaes
            if rm_edge_obj:
                if minX==0 or minY==0 or maxX==(width-1) or maxY==(height-1):
                    print('warning, object (minX, minY, maxX, maxY): (%d %d %d %d) touched the edge in %s, ignore it'%
                          (minX, minY, maxX, maxY, label_path))
                    continue

            # in semantic, class_id 0 is background, yolo, class 0 is target, so minus 1
            class_id -= 1
            x, y, w, h = convert((width,height), (minX, maxX, minY, maxY))
            f_obj.writelines('%d %f %f %f %f\n'%(class_id, x, y, w, h))

            ## output objects to polygons for checking
            if b_save_vector:
                geo_transform = raster_io.get_transform_from_file(image_path)
                geo_xs, geo_ys = raster_io.pixel_xy_to_geo_xy_list([minY,maxY],[minX,maxX],geo_transform)
                obj_polygon = vector_gpd.convert_image_bound_to_shapely_polygon([geo_xs[0],geo_ys[0], geo_xs[1], geo_ys[1]])   # (left, bottom, right, top)
                obj_polygons.append(obj_polygon)

        # save objects to a vector file for checking
        if len(obj_polygons) > 0 and b_save_vector:
            save_pd = pd.DataFrame({'Polygon': obj_polygons})
            ref_prj = map_projection.get_raster_or_vector_srs_info_proj4(image_path)
            save_path_gpkg = os.path.splitext(image_path)[0] + '.gpkg'
            vector_gpd.save_polygons_to_files(save_pd,'Polygon', ref_prj,save_path_gpkg, format='GPKG')

def get_image_list(txt_dir,sample_txt,img_dir, img_ext):
    img_list = []
    with open(os.path.join(txt_dir,sample_txt), 'r') as f_obj:
        lines = [item.strip() for item in  f_obj.readlines()]
        for line in lines:
            img_list.append(os.path.join(img_dir, line + img_ext))
    return img_list

def image_label_to_yolo_format(para_file):

    print("Image labels (semantic segmentation) to YOLO object detection")

    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    img_ext = parameters.get_string_parameters_None_if_absence(para_file,'split_image_format')
    proc_num = parameters.get_digit_parameters(para_file, 'process_num', 'int')

    SECONDS = time.time()

    # get image and label path
    image_list = []
    label_list = []
    with open(os.path.join('list','trainval.txt'), 'r') as f_obj:
        lines = [item.strip() for item in  f_obj.readlines()]
        for line in lines:
            image_list.append(os.path.join('split_images', line + img_ext))
            label_list.append(os.path.join('split_labels', line + img_ext))

    num_classes_noBG = parameters.get_digit_parameters_None_if_absence(para_file, 'NUM_CLASSES_noBG', 'int')
    b_ignore_edge_objects = parameters.get_bool_parameters_None_if_absence(para_file,'b_ignore_edge_objects')
    if b_ignore_edge_objects is None:
        b_ignore_edge_objects = False

    b_save_objects_to_vector = parameters.get_bool_parameters_None_if_absence(para_file,'b_save_objects_to_vector')
    if b_save_objects_to_vector is None:
        b_save_objects_to_vector = False

    # get boxes
    total_count = len(image_list)
    for idx, (img, label) in enumerate(zip(image_list,label_list)):
        get_yolo_boxes_one_img(idx, total_count, img, label,num_classes_noBG,rm_edge_obj=b_ignore_edge_objects, b_save_vector=b_save_objects_to_vector)

    # write obj.data file
    train_sample_txt = parameters.get_string_parameters(para_file, 'training_sample_list_txt')
    val_sample_txt = parameters.get_string_parameters(para_file, 'validation_sample_list_txt')
    train_img_list = get_image_list('list',train_sample_txt,'split_images',img_ext)
    val_img_list = get_image_list('list',val_sample_txt,'split_images',img_ext)

    expr_name = parameters.get_string_parameters(para_file,'expr_name')
    object_names = parameters.get_string_list_parameters(para_file,'object_names')
    io_function.mkdir('data')
    io_function.mkdir(expr_name)

    with open(os.path.join('data','obj.data'), 'w') as f_obj:
        f_obj.writelines('classes = %d'%num_classes_noBG + '\n')

        train_txt = os.path.join('data','train.txt')
        io_function.save_list_to_txt(train_txt,train_img_list)
        f_obj.writelines('train = %s'%train_txt+ '\n')

        val_txt = os.path.join('data','val.txt')
        io_function.save_list_to_txt(val_txt, val_img_list)
        f_obj.writelines('valid = %s' % val_txt + '\n')

        obj_name_txt = os.path.join('data','obj.names')
        io_function.save_list_to_txt(obj_name_txt,object_names)
        f_obj.writelines('names = %s' % obj_name_txt + '\n')

        f_obj.writelines('backup = %s'%expr_name + '\n')


    duration = time.time() - SECONDS
    os.system('echo "$(date): time cost of converting to yolo format: %.2f seconds">>time_cost.txt' % duration)

    pass



def main(options, args):
    para_file= args[0]
    image_label_to_yolo_format(para_file)


if __name__ == '__main__':
    usage = "usage: %prog [options] para_file "
    parser = OptionParser(usage=usage, version="1.0 2021-04-04")
    parser.description = 'Introduction: convert split images and labels to yolo format (objection) '

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)