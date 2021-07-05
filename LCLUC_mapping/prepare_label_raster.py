#!/usr/bin/env python
# Filename: prepare_label_raster.py 
"""
introduction: for test, crop and resample label raster for training.

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 05 July, 2021
"""

import os,sys

code_dir = os.path.expanduser('~/codes/PycharmProjects/DeeplabforRS')
sys.path.insert(0, code_dir)

import basic_src.io_function as io_function
import raster_io
import basic_src.RSImageProcess as RSImageProcess
import basic_src.map_projection as map_projection

data_dir=os.path.expanduser('~/Data/LandCover_LandUse_Change')


def resample_crop_raster(ref_raster, input_raster, output_raster=None, resample_method='near'):

    if output_raster is None:
        output_raster = io_function.get_name_by_adding_tail(os.path.basename(input_raster),'res_sub')

    if os.path.isfile(output_raster):
        print('Warning, %s exists'%output_raster)
        return output_raster

    # check projection
    prj4_ref = map_projection.get_raster_or_vector_srs_info_proj4(ref_raster)
    prj4_input = map_projection.get_raster_or_vector_srs_info_proj4(input_raster)
    if prj4_ref != prj4_input:
        raise ValueError('projection inconsistent: %s and %s'%(ref_raster, input_raster))

    # crop
    RSImageProcess.subset_image_baseimage(output_raster, input_raster, ref_raster, same_res=True,resample_m=resample_method)
    if os.path.isfile(output_raster):
        return output_raster
    else:
        return False

def crop_resample_label_raster():
    img_path = os.path.join(data_dir,'rs_imagery/Planet/Brazil_area1_2019Feb07_psscene4band_analytic_sr_udm2/Brazil_area1_20190207_3B_AnalyticMS_SR_mosaic_8bit_rgb.tif')

    label_path = os.path.join(data_dir, 'LCLUC_MapBiomas_Gabriel/COLECAO_5_DOWNLOADS_COLECOES_ANUAL_2019_merge.tif')

    # crop and resample
    resample_crop_raster(img_path,label_path)



def main():
    crop_resample_label_raster()

if __name__ == '__main__':
    main()
    pass