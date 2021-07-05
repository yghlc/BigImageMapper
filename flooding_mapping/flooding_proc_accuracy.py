#!/usr/bin/env python
# Filename: flooding_proc_accuracy.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 04 July, 2021
"""

import os,sys
machine_name = os.uname()[1]

import numpy as np


code_dir = os.path.expanduser('~/codes/PycharmProjects/DeeplabforRS')
sys.path.insert(0, code_dir)

import basic_src.io_function as io_function
import raster_io
import basic_src.RSImageProcess as RSImageProcess

code_dir2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir2)

import utility.classify_assess as classify_assess




if machine_name == 'ubuntu':
    data_dir = os.path.expanduser('~/Bhaltos2/lingcaoHuang/flooding_area')
    sur_water_dir = os.path.expanduser('~/Bhaltos2/lingcaoHuang/global_surface_water')
elif machine_name=='donostia':
    data_dir = os.path.expanduser('~/Bhaltos2/flooding_area')
    sur_water_dir = os.path.expanduser('~/Bhaltos2/global_surface_water')
else:
    data_dir = os.path.expanduser('~/Data/flooding_area')
    sur_water_dir = os.path.expanduser('~/Data/global_surface_water')


def resample_crop_raster(ref_raster, input_raster, output_raster=None, resample_method='near'):

    if output_raster is None:
        output_raster = io_function.get_name_by_adding_tail(os.path.basename(input_raster),'res_sub')

    # xres, yres = raster_io.get_xres_yres_file(ref_raster)
    # resample_raster = os.path.basename(input_raster)
    # resample_raster = io_function.get_name_by_adding_tail(resample_raster,'resample')
    #
    # # resample
    # RSImageProcess.resample_image(input_raster,resample_raster,xres,yres,resample_method)
    # if os.path.isfile(resample_raster) is False:
    #     raise ValueError('Resample %s failed'%input_raster)

    if os.path.isfile(output_raster):
        print('Warning, %s exists'%output_raster)
        return output_raster

    # crop
    RSImageProcess.subset_image_baseimage(output_raster, input_raster, ref_raster, same_res=True,resample_m=resample_method)
    if os.path.isfile(output_raster):
        return output_raster
    else:
        return False

def mask_by_surface_water(map_raster, surface_water_crop):

    # save mask result to current folder
    save_mask_result = io_function.get_name_by_adding_tail(os.path.basename(map_raster),'WaterMask')
    if os.path.isfile(save_mask_result):
        print('warning, %s exists'%save_mask_result)
        return save_mask_result

    # read
    map_array_2d,nodata = raster_io.read_raster_one_band_np(map_raster)
    water_array_2d,_ = raster_io.read_raster_one_band_np(surface_water_crop)

    print(map_array_2d.shape)
    if map_array_2d.shape != water_array_2d.shape:
        raise ValueError('size inconsistent: %s and %s'%(str(map_array_2d.shape), str(water_array_2d.shape)))

    # mask out pixel, original is water or others
    map_array_2d[ np.logical_or(water_array_2d==1, water_array_2d==255) ] = 0

    if raster_io.save_numpy_array_to_rasterfile(map_array_2d,save_mask_result,map_raster, compress='lzw',tiled='Yes',bigtiff='if_safer'):
        return save_mask_result


def post_processing_Houston():
    # mapping raster
    dl_map_20170829 = os.path.join(data_dir,
    'mapping_polygons_rasters/exp1_grd_Houston/Houston_SAR_20170829_houston_deeplabV3+_1_exp1_post_1_label.tif')

    # other map results (consider as ground truth)
    modis_map_20170829 = os.path.join(data_dir,
    'flooding_rasters_other_results/Houston/2017082920170805_cleaned_region_prj_label.tif')

    # global surface water
    surface_water=os.path.join(sur_water_dir,'extent_epsg2163_houston/extent_100W_30_40N_v1_3_2020.tif')

    # elevation (SRTM)
    # dem_path = os.path.join(data_dir,'DEM/Houston/Houston_SRTM_prj.tif')

    # resample and crop to the same resolution and extent
    surface_water_crop = resample_crop_raster(dl_map_20170829,surface_water)
    # dem_crop = resample_crop_raster(dl_map_20170829,dem_path)

    # mask by using the global surface water
    dl_map_20170829_watermask = mask_by_surface_water(dl_map_20170829,surface_water_crop)

    modis_map_20170829_watermask = mask_by_surface_water(modis_map_20170829,surface_water_crop)

    # condauct accuracy assesement
    classify_assess.pixel_accuracy_assessment(modis_map_20170829_watermask,dl_map_20170829_watermask)


    pass

def post_processing_Goapara():

    # mapping raster
    dl_map_20200516 = os.path.join(data_dir,'mapping_polygons_rasters/exp1_grd_Goalpara/Goalpara_SAR_20200516_houston_deeplabV3+_1_exp1_post_1_label.tif')
    dl_map_20200516 = os.path.join(data_dir,'mapping_polygons_rasters/exp1_grd_Goalpara/Goalpara_SAR_20200528_houston_deeplabV3+_1_exp1_post_1_label.tif')
    dl_map_20200516 = os.path.join(data_dir,'mapping_polygons_rasters/exp1_grd_Goalpara/Goalpara_SAR_20200609_houston_deeplabV3+_1_exp1_post_1_label.tif')
    dl_map_20200516 = os.path.join(data_dir,'mapping_polygons_rasters/exp1_grd_Goalpara/Goalpara_SAR_20200621_houston_deeplabV3+_1_exp1_post_1_label.tif')
    dl_map_20200516 = os.path.join(data_dir,'mapping_polygons_rasters/exp1_grd_Goalpara/Goalpara_SAR_20200703_houston_deeplabV3+_1_exp1_post_1_label.tif')
    dl_map_20200516 = os.path.join(data_dir,'mapping_polygons_rasters/exp1_grd_Goalpara/Goalpara_SAR_20200715_houston_deeplabV3+_1_exp1_post_1_label.tif')
    dl_map_20200516 = os.path.join(data_dir,'mapping_polygons_rasters/exp1_grd_Goalpara/Goalpara_SAR_20200727_houston_deeplabV3+_1_exp1_post_1_label.tif')
    dl_map_20200516 = os.path.join(data_dir,'mapping_polygons_rasters/exp1_grd_Goalpara/Goalpara_SAR_20200808_houston_deeplabV3+_1_exp1_post_1_label.tif')
    dl_map_20200516 = os.path.join(data_dir,'mapping_polygons_rasters/exp1_grd_Goalpara/Goalpara_SAR_20200820_houston_deeplabV3+_1_exp1_post_1_label.tif')

    # other map results (consider as ground truth)
    modis_map_20200705 = os.path.join(data_dir,'flooding_rasters_other_results/Goapara/20200705MODIS_region_prj_label.tif')

    # global surface water
    surface_water = os.path.join(sur_water_dir,'extent_UTM46N_Goalpara/extent_80_90E_30N_v1_3_2020.tif')

    # elevation (SRTM)
    dem_path = os.path.join(data_dir, 'extent_UTM46N_Goalpara/extent_80_90E_30N_v1_3_2020.tif')

    # resample and crop to the same resolution and extent


    # post-processing the multi-temporal results


    # mask by using the global surface water

    pass

def main():

    post_processing_Houston()


    pass


if __name__ == '__main__':
    main()
    pass