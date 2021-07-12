#!/usr/bin/env python
# Filename: flooding_proc_accuracy.py 
"""
introduction: processing the mapping results and caluculate the accuracy

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 12 July, 2021
"""

import os,sys
machine_name = os.uname()[1]

import numpy as np


code_dir = os.path.expanduser('~/codes/PycharmProjects/DeeplabforRS')
sys.path.insert(0, code_dir)

import basic_src.io_function as io_function
import raster_io
import basic_src.RSImageProcess as RSImageProcess
import basic_src.map_projection as map_projection

code_dir2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir2)

import utility.classify_assess as classify_assess


data_dir = os.path.expanduser('~/Data/LandCover_LandUse_Change')


def resample_crop_raster(ref_raster, input_raster, output_raster=None, resample_method='near'):

    if output_raster is None:
        output_raster = io_function.get_name_by_adding_tail(os.path.basename(input_raster),'res_sub')


    # check projection
    prj4_ref = map_projection.get_raster_or_vector_srs_info_proj4(ref_raster)
    prj4_input = map_projection.get_raster_or_vector_srs_info_proj4(input_raster)
    if prj4_ref != prj4_input:
        raise ValueError('projection inconsistent: %s and %s'%(ref_raster, input_raster))

    if os.path.isfile(output_raster):
        print('Warning, %s exists'%output_raster)
        return output_raster

    # crop
    RSImageProcess.subset_image_baseimage(output_raster, input_raster, ref_raster, same_res=True,resample_m=resample_method)
    if os.path.isfile(output_raster):
        return output_raster
    else:
        return False



def post_processing_area1():
    # mapping raster
    area1_map_20190207 = os.path.join(data_dir,
    'automapping/brazil_deeplabv3+_1/result_backup/Brazil_area1_rgb_20190207_brazil_deeplabv3+_1_exp1_1/I0_Brazil_area1_rgb_20190207_brazil_deeplabv3+_1_exp1.tif')

    # LCLU from MapBiomas (consider as ground truth)
    MapBiomas_map_2019 = os.path.join(data_dir,
    'LCLUC_MapBiomas_Gabriel/COLECAO_5_DOWNLOADS_COLECOES_ANUAL_2019_merge_prj_crop.tif')


    gt_raster = resample_crop_raster(area1_map_20190207,MapBiomas_map_2019)

    nodata = 0

    # condauct accuracy assesement
    classify_assess.pixel_accuracy_assessment(gt_raster,area1_map_20190207,no_data=nodata)


    pass

def post_processing_area1_extend():
    # mapping raster
    area1_map_20190207 = os.path.join(data_dir,
    'automapping/brazil_deeplabv3+_1/result_backup/Brazil_area1_extend_rgb_20190207_brazil_deeplabv3+_1_exp1_1/I0_Brazil_area1_extend_rgb_20190207_brazil_deeplabv3+_1_exp1.tif')

    # LCLU from MapBiomas (consider as ground truth)
    MapBiomas_map_2019 = os.path.join(data_dir,
    'LCLUC_MapBiomas_Gabriel/COLECAO_5_DOWNLOADS_COLECOES_ANUAL_2019_merge_prj_crop.tif')

    output_gr_raster = 'COLECAO_5_DOWNLOADS_COLECOES_ANUAL_2019_merge_prj_crop_for_area1_extend.tif'
    gt_raster = resample_crop_raster(area1_map_20190207,MapBiomas_map_2019,output_raster=output_gr_raster)
    nodata = 0
    # condauct accuracy assesement
    classify_assess.pixel_accuracy_assessment(gt_raster,area1_map_20190207,no_data=nodata)


def main():

    # post_processing_area1()
    post_processing_area1_extend()

    pass


if __name__ == '__main__':
    main()
    pass