#!/usr/bin/env python
# Filename: prepare_data_for_webValidtion.py 
"""
introduction: prepare data for web validation

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 03 May, 2022
"""

import os, sys
code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)

import re

import basic_src.io_function as io_function
import basic_src.basic as basic

poly2geojson = os.path.expanduser('~/codes/PycharmProjects/Landuse_DL/web-validation/polygons2geojson.py')
imgExt = os.path.expanduser('~/codes/PycharmProjects/rs_data_proc/tools/get_images_extent_shp.py')

def polygons2geojson(shp_path, save_dir):
    command_str =   poly2geojson + ' ' +   shp_path + ' ' + save_dir
    print(command_str)
    basic.os_system_exit_code(command_str)

def tifs_to_png(image_dir):
    tif_list = io_function.get_file_list_by_pattern(image_dir,'*/*.tif')
    for idx, tif in enumerate(tif_list):
        print('tif to png: %d/%d tif'%(idx+1,len(tif_list)))
        basename = io_function.get_name_no_ext(tif)
        save_path = os.path.join(image_dir,basename+'.png')
        if os.path.isfile(save_path):
            print('%s exists, skip'%save_path)
            continue
        command_str = "gdal_translate -of PNG %s %s"%(tif,save_path)
        basic.os_system_exit_code(command_str)

def get_tifs_bounding_boxes(image_dir):
    tif_list = io_function.get_file_list_by_pattern(image_dir,'*/*.tif')
    for idx, tif in enumerate(tif_list):
        print('get bounding box: %d/%d tif'%(idx+1,len(tif_list)))
        basename = io_function.get_name_no_ext(tif)
        save_path = os.path.join(image_dir, basename + '_bound.geojson')
        if os.path.isfile(save_path):
            print('%s exists, skip'%save_path)
            continue

        command_str = imgExt + " %s -o tmp.gpkg" %tif
        basic.os_system_exit_code(command_str)
        command_str = "ogr2ogr -f GeoJSON -t_srs EPSG:3413 %s tmp.gpkg"%save_path  # note: projection is EPSG:3413
        basic.os_system_exit_code(command_str)

        io_function.delete_file_or_dir('tmp.gpkg')

def organize_files(sub_img_dirs, save_dir):
    if os.path.isfile(save_dir) is False:
        io_function.mkdir(save_dir)

    # get all png files
    png_list = []
    for img_dir in sub_img_dirs:
        pngs = io_function.get_file_list_by_pattern(img_dir,'*.png')
        png_list.extend(pngs)

    image_name_list = []
    images_dir = os.path.join(save_dir,'images')
    imageBound_dir = os.path.join(save_dir,'imageBound')
    objectPolygons_dir = os.path.join(save_dir,'objectPolygons')
    io_function.mkdir(images_dir)
    io_function.mkdir(imageBound_dir)
    io_function.mkdir(objectPolygons_dir)

    for idx, png in enumerate(png_list):
        basename = io_function.get_name_no_ext(png)
        new_name = 'img'+str(idx+1).zfill(6) + '_' + basename
        image_name_list.append(new_name)

        io_function.copy_file_to_dst(png,os.path.join(images_dir,new_name+'.png'))
        png_xml = png + '.aux.xml'
        if os.path.isfile(png_xml):
            io_function.copy_file_to_dst(png_xml, os.path.join(images_dir, new_name + '.png.aux.xml'))

        bound_path = png.replace('.png','_bound.geojson')
        io_function.copy_file_to_dst(bound_path,os.path.join(imageBound_dir,new_name+'_bound.geojson'))

        digit_str = re.findall(r'_\d+', basename)
        id_str = digit_str[0][1:]
        object_path = os.path.join(os.path.dirname(png), 'id_%s.geojson'%id_str)
        io_function.copy_file_to_dst(object_path, os.path.join(objectPolygons_dir, new_name + '.geojson'))

    txt_path = os.path.join(save_dir,'imageList.txt')
    io_function.save_list_to_txt(txt_path,image_name_list)

def alaska_main():
    # working on tesia:
    work_dir = os.path.expanduser('~/Data/Arctic/alaska/sub_images_for_web_validate')

    shp_dir = os.path.expanduser('~/Data/temp/alaskaNS_yolov4_5/result_backup')
    shp1 = os.path.join(shp_dir,'alaNS_hillshadeHWline_2008to2017_alaskaNS_yolov4_5_exp5_1/'
                                'alaNS_hillshadeHWline_2008to2017_alaskaNS_yolov4_5_exp5_post_1_select_manuSelect.shp')
    shp2 = os.path.join(shp_dir,'alaskaNotNS_hillshadeHWline_2008to2017_alaskaNS_yolov4_5_exp5_1/'
                                'alaskaNotNS_hillshadeHWline_2008to2017_alaskaNS_yolov4_5_exp5_post_1_select_manuSelect.shp')

    # step 1: get subimages
    # using ~/codes/PycharmProjects/ChangeDet_DL/dataTools/extract_subimage_timeSeries.py to get sub-image, but only one image
    # see more in "get_multi_sub_images.sh" in shp1_subImg_dir and shp2_subImg_dir
    shp1_subImg_dir = os.path.join(work_dir,'alaNS_yolov4_5_exp5_1_manu_sel')
    shp2_subImg_dir = os.path.join(work_dir,'alaskaNotNS_yolov4_5_exp5_1_manu_sel')

    # step 2: get geojson for each polygons
    polygons2geojson(shp1,shp1_subImg_dir)
    polygons2geojson(shp2,shp2_subImg_dir)

    # step 3: convert subimages in the format of geotiff to pngs
    tifs_to_png(shp1_subImg_dir)
    tifs_to_png(shp2_subImg_dir)

    # step 4: get bounding box of sub-images
    get_tifs_bounding_boxes(shp1_subImg_dir)
    get_tifs_bounding_boxes(shp2_subImg_dir)

    # step 5: organize files
    organize_files([shp1_subImg_dir,shp2_subImg_dir], os.path.join(work_dir,'data'))


def panArctic_main():
    # working on tesia:
    work_dir = os.path.expanduser('~/Data/Arctic/pan_Arctic/sub_images_for_web_validate')

    shp_dir = os.path.expanduser('~/Data/Arctic/pan_Arctic/arcticdem_results_thawslump')
    shp1 = os.path.join(shp_dir,'panArctic_thawSlump_from_ArcticDEM_v2.shp')

    # step 1: get subimages
    # using ~/codes/PycharmProjects/ChangeDet_DL/dataTools/extract_subimage_timeSeries.py to get sub-image, but only one image
    # see more in "get_multi_sub_images.sh" in shp1_subImg_dir and shp2_subImg_dir
    shp1_subImg_dir = os.path.join(work_dir,'panArctic_thawSlump_from_ArcticDEM_v2')

    # step 2: get geojson for each polygons
    polygons2geojson(shp1,shp1_subImg_dir)

    # step 3: convert subimages in the format of geotiff to pngs
    tifs_to_png(shp1_subImg_dir)

    # step 4: get bounding box of sub-images
    get_tifs_bounding_boxes(shp1_subImg_dir)

    # step 5: organize files
    organize_files([shp1_subImg_dir], os.path.join(work_dir,'data'))

def main():

    # alaska_main()
    panArctic_main()


    pass

if __name__ == '__main__':
    main()
    pass