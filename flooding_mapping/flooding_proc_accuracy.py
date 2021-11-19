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
import basic_src.map_projection as map_projection

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


def resample_crop_raster_using_shp(ref_shp, input_raster, output_raster=None, resample_method='near'):
    if output_raster is None:
        output_raster = io_function.get_name_by_adding_tail(os.path.basename(input_raster),'res_sub')

    # check projection
    prj4_ref = map_projection.get_raster_or_vector_srs_info_proj4(ref_shp)
    prj4_input = map_projection.get_raster_or_vector_srs_info_proj4(input_raster)
    if prj4_ref != prj4_input:
        raise ValueError('projection inconsistent: %s and %s'%(ref_shp, input_raster))

    if os.path.isfile(output_raster):
        print('Warning, %s exists'%output_raster)
        return output_raster

    # crop
    out_res = 10
    # RSImageProcess.subset_image_baseimage(output_raster, input_raster, ref_raster, same_res=True,resample_m=resample_method)
    RSImageProcess.subset_image_by_shapefile(input_raster,ref_shp,save_path=output_raster, dst_nondata=128,resample_m=resample_method,
                                             xres=out_res, yres=out_res,compress='lzw', tiled='yes', bigtiff='IF_SAFER')
    if os.path.isfile(output_raster):
        return output_raster
    else:
        return False


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

def mask_by_surface_water(map_raster, surface_water_crop, save_dir='./'):

    # save mask result to current folder
    save_mask_result = io_function.get_name_by_adding_tail(os.path.basename(map_raster),'WaterMask')
    save_mask_result = os.path.join(save_dir,save_mask_result)
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

def mask_by_elevation(map_raster_path, elevation_crop_path, threashold):

    # save mask result to current folder
    save_mask_result = io_function.get_name_by_adding_tail(os.path.basename(map_raster_path),'DEMMask')
    if os.path.isfile(save_mask_result):
        print('warning, %s exists'%save_mask_result)
        return save_mask_result

    # read
    map_array_2d,nodata = raster_io.read_raster_one_band_np(map_raster_path)
    dem_array_2d,_ = raster_io.read_raster_one_band_np(elevation_crop_path)

    print(map_array_2d.shape)
    if map_array_2d.shape != dem_array_2d.shape:
        raise ValueError('size inconsistent: %s and %s'%(str(map_array_2d.shape), str(dem_array_2d.shape)))

    # mask out pixel with high elevation
    map_array_2d[ dem_array_2d > threashold ] = 0

    if raster_io.save_numpy_array_to_rasterfile(map_array_2d,save_mask_result,map_raster_path, compress='lzw',tiled='Yes',bigtiff='if_safer'):
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
    save_dir = os.path.dirname(dl_map_20170829)
    dl_map_20170829_watermask = mask_by_surface_water(dl_map_20170829,surface_water_crop,save_dir=save_dir)

    modis_map_20170829_watermask = mask_by_surface_water(modis_map_20170829,surface_water_crop)

    # condauct accuracy assesement
    # classify_assess.pixel_accuracy_assessment(modis_map_20170829_watermask,dl_map_20170829_watermask)


def apply_water_mask_to_mapping_result_Houston():

    #  these three mapping results have the same extent
    dl_map_results = ['mapping_polygons_rasters/exp1_grd_Houston/Houston_SAR_20170829_houston_deeplabV3+_1_exp1_post_1_label.tif',
                      'mapping_polygons_rasters/exp2_binary_Houston/Houston_SAR_20170829_houston_deeplabV3+_1_exp2_post_1_label.tif',
                      'mapping_polygons_rasters/exp4_3band_Houston/Houston_SAR_polar_20170829_houston_deeplabV3+_1_exp4_post_2_label.tif']

    dl_map_results = [os.path.join(data_dir,item) for item in dl_map_results]
    # global surface water
    surface_water = os.path.join(sur_water_dir, 'extent_epsg2163_houston/extent_100W_30_40N_v1_3_2020.tif')

    # resample and crop to the same resolution and extent
    surface_water_crop = resample_crop_raster(dl_map_results[0],surface_water)
    for dl_map_res in dl_map_results:
        save_dir = os.path.dirname(dl_map_res)
        dl_map_res_watermask = mask_by_surface_water(dl_map_res, surface_water_crop, save_dir=save_dir)


    pass

def post_processing_Goapara():

    # mapping raster
    # these mapping raster have the same width and height: Size is 28825, 21970
    dl_map_20200516 = os.path.join(data_dir,'mapping_polygons_rasters/exp1_grd_Goalpara/Goalpara_SAR_20200516_houston_deeplabV3+_1_exp1_post_1_label.tif')
    # dl_map_20200528 = os.path.join(data_dir,'mapping_polygons_rasters/exp1_grd_Goalpara/Goalpara_SAR_20200528_houston_deeplabV3+_1_exp1_post_1_label.tif')
    # dl_map_20200609 = os.path.join(data_dir,'mapping_polygons_rasters/exp1_grd_Goalpara/Goalpara_SAR_20200609_houston_deeplabV3+_1_exp1_post_1_label.tif')
    # dl_map_20200621 = os.path.join(data_dir,'mapping_polygons_rasters/exp1_grd_Goalpara/Goalpara_SAR_20200621_houston_deeplabV3+_1_exp1_post_1_label.tif')
    # dl_map_20200703 = os.path.join(data_dir,'mapping_polygons_rasters/exp1_grd_Goalpara/Goalpara_SAR_20200703_houston_deeplabV3+_1_exp1_post_1_label.tif')
    # dl_map_20200715 = os.path.join(data_dir,'mapping_polygons_rasters/exp1_grd_Goalpara/Goalpara_SAR_20200715_houston_deeplabV3+_1_exp1_post_1_label.tif')
    # dl_map_20200727 = os.path.join(data_dir,'mapping_polygons_rasters/exp1_grd_Goalpara/Goalpara_SAR_20200727_houston_deeplabV3+_1_exp1_post_1_label.tif')
    # dl_map_20200808 = os.path.join(data_dir,'mapping_polygons_rasters/exp1_grd_Goalpara/Goalpara_SAR_20200808_houston_deeplabV3+_1_exp1_post_1_label.tif')
    # dl_map_20200820 = os.path.join(data_dir,'mapping_polygons_rasters/exp1_grd_Goalpara/Goalpara_SAR_20200820_houston_deeplabV3+_1_exp1_post_1_label.tif')

    # get dl map results (include dl_map_20200516)
    dl_map_results = io_function.get_file_list_by_ext('.tif',os.path.join(data_dir,'mapping_polygons_rasters/exp1_grd_Goalpara'), bsub_folder=False)

    # other map results (consider as ground truth)
    modis_map_20200705 = os.path.join(data_dir,'flooding_rasters_other_results/Goapara/20200705MODIS_region_prj_label.tif')

    # global surface water
    surface_water = os.path.join(sur_water_dir,'extent_UTM46N_Goalpara/extent_80_90E_30N_v1_3_2020.tif')

    # elevation (SRTM)
    dem_path = os.path.join(data_dir, 'DEM/Goalpara/Goalpara_SRTM_prj.tif')

    # resample and crop to the same resolution and extent
    surface_water_crop = resample_crop_raster(dl_map_20200516,surface_water)
    dem_crop = resample_crop_raster(dl_map_20200516,dem_path)

    # mask by using the global surface water
    dl_map_results_watermask = [mask_by_surface_water(item,surface_water_crop) for item in dl_map_results ]
    modis_map_20200705_watermask = mask_by_surface_water(modis_map_20200705, surface_water_crop)
    dl_map_results = dl_map_results_watermask

    # post-processing by using the DEM
    elevation_thr = 500
    dl_map_results_mask = [mask_by_elevation(item, dem_crop,elevation_thr) for item in dl_map_results]

    # post-processing the multi-temporal results

    # calculate accuracy for each dl mapping results
    for idx, dl_map in enumerate(dl_map_results_mask):
        print(idx, dl_map)
        classify_assess.pixel_accuracy_assessment(modis_map_20200705_watermask,dl_map)


    pass

def main():

    # post_processing_Houston()

    apply_water_mask_to_mapping_result_Houston()

    # post_processing_Goapara()


    pass


if __name__ == '__main__':
    main()
    pass