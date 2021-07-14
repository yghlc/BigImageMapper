#!/usr/bin/env python
# Filename: get_color_map_for_raster_inQGIS.py 
"""
introduction: get color table file for raster in QGIS

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 13 July, 2021
"""

import os,sys

code_dir = os.path.expanduser('~/codes/PycharmProjects/DeeplabforRS')
sys.path.insert(0, code_dir)
import raster_io
import basic_src.io_function as io_function

import numpy as np

def get_new_color_table_for_raster(raster, color_table_txt, out_dir=None):
    data,no_data = raster_io.read_raster_one_band_np(raster)

    # remove nodata
    data_1d = data.flatten()
    print(data_1d.shape)
    data_1d = data_1d[data_1d != no_data]
    print(data_1d.shape)
    unique_values = np.unique(data_1d)
    print('unique_values:', unique_values)

    save_color_table = io_function.get_name_no_ext(os.path.basename(raster)) + '_color.txt'
    if out_dir is not None:
        save_color_table = os.path.join(out_dir,save_color_table)

    save_lines = []
    with open(color_table_txt, 'r') as f_obj:
        all_lines = f_obj.readlines()

        # copy first two lines
        save_lines.append(all_lines[0])
        save_lines.append(all_lines[1])

        for idx in range(2,len(all_lines)):
            value = int(all_lines[idx].split(',')[0])
            if value in unique_values:
                save_lines.append(all_lines[idx])

    with open(save_color_table,'w') as f_obj:
        f_obj.writelines(save_lines)

    print('Save color table to %s'%os.path.abspath(save_color_table))





def main():
    LCLU_mapbiomas_color = os.path.expanduser('~/Data/qgis_files/LCLU_mapbiomas.txt')

    map_raster = "/Users/huanglingcao/Data/LandCover_LandUse_Change/automapping/" \
                 "brazil_deeplabv3+_1/result_backup/Brazil_area1_rgb_20190207_brazil_deeplabv3+_1_exp1_1/" \
                 "I0_Brazil_area1_rgb_20190207_brazil_deeplabv3+_1_exp1.tif"

    map_raster_extend = "/Users/huanglingcao/Data/LandCover_LandUse_Change/automapping/brazil_deeplabv3+_1/result_backup/" \
                        "Brazil_area1_extend_rgb_20190207_brazil_deeplabv3+_1_exp1_1/" \
                        "I0_Brazil_area1_extend_rgb_20190207_brazil_deeplabv3+_1_exp1.tif"

    map_raster_landsat = "/Users/huanglingcao/codes/PycharmProjects/Landuse_DL/LCLUC_mapping/" \
                         "COLECAO_5_DOWNLOADS_COLECOES_ANUAL_2019_merge_prj_crop_for_area1_extend.tif"

    map_raster_landsat_2019_crop = "/Users/huanglingcao/Data/LandCover_LandUse_Change/LCLUC_MapBiomas_Gabriel/" \
    "COLECAO_5_DOWNLOADS_COLECOES_ANUAL_2019_merge_prj_crop.tif"

    save_dir = os.path.expanduser('~/Data/LandCover_LandUse_Change/LCLUC_MapBiomas_Gabriel')

    # get_new_color_table_for_raster(map_raster, LCLU_mapbiomas_color, out_dir=save_dir)
    # get_new_color_table_for_raster(map_raster_extend, LCLU_mapbiomas_color, out_dir=save_dir)
    # get_new_color_table_for_raster(map_raster_landsat, LCLU_mapbiomas_color, out_dir=save_dir)
    get_new_color_table_for_raster(map_raster_landsat_2019_crop, LCLU_mapbiomas_color, out_dir=save_dir)


if __name__ == '__main__':
    main()