#!/usr/bin/env python
# Filename: crop_resample_surface_water.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 09 November, 2021
"""

import os,sys

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

if machine_name == 'ubuntu':
    data_dir = os.path.expanduser('~/Bhaltos2/lingcaoHuang/flooding_area')
    sur_water_dir = os.path.expanduser('~/Bhaltos2/lingcaoHuang/global_surface_water')
elif machine_name=='donostia':
    data_dir = os.path.expanduser('~/Bhaltos2/flooding_area')
    sur_water_dir = os.path.expanduser('~/Bhaltos2/global_surface_water')
else:
    data_dir = os.path.expanduser('~/Data/flooding_area')
    sur_water_dir = os.path.expanduser('~/Data/global_surface_water')


from flooding_proc_accuracy import resample_crop_raster
from flooding_proc_accuracy import resample_crop_raster_using_shp


def mask_nodata_regions_surface_water(ref_raster, in_raster, out_raster, ref_nodata=0, out_nodata=0):

    if os.path.isfile(out_raster):
        print('%s already exists'%out_raster)
        return True

    # get nodata mask from ref_raster
    command_str = 'gdal_calc.py --calc="(A>%d)" --outfile=tmp.tif -A %s --NoDataValue 0 --type=Byte '%(ref_nodata,ref_raster)

    res = os.system(command_str)
    if res != 0:
        print(res)
        sys.exit(1)

    # apply the mask to in_raster
    # in the surface water, 1 is water, 0 are other, so mask water outside extent to zero as well, eventually, set 0 as nodata
    command_str = 'gdal_calc.py --calc="B*A" --outfile=%s -A tmp.tif -B %s --NoDataValue %d --type=Byte ' % (out_raster,in_raster,out_nodata)

    res = os.system(command_str)
    if res != 0:
        print(res)
        sys.exit(1)

    io_function.delete_file_or_dir('tmp.tif')

    return True

def surface_water_Houston_use_shapefile():

    # notes that: these is a tiny gap between ref_raster_up and ref_raster_down, around 30 m width,
    # end in a line in the figure of surface water.
    # think about using another way to crop surface water, get shapefile of these two, then edit the shapefile and use the shapefile for cropping

    ref_raster_up = os.path.join(data_dir,
    'Houston/Houston_SAR_GRD_FLOAT_gee/S1_Houston_prj_8bit_select/S1A_IW_GRDH_1SDV_20170829T002645_20170829T002710_018131_01E74D_3220_prj_8bit.tif')

    ref_raster_down = os.path.join(data_dir,
    'Houston/Houston_SAR_GRD_FLOAT_gee/S1_Houston_prj_8bit_select/S1A_IW_GRDH_1SDV_20170829T002620_20170829T002645_018131_01E74D_D734_prj_8bit.tif')


    # # get shapefile
    # py = os.path.expanduser('~/codes/PycharmProjects/rs_data_proc/tools/get_image_valid_extent.py')
    # command_str = py + '  ' + ref_raster_up
    # res = os.system(command_str)
    # if res != 0:
    #     sys.exit(1)
    #
    # command_str = py + '  ' + ref_raster_down
    # res = os.system(command_str)
    # if res != 0:
    #     sys.exit(1)

    # based the merge and edit the shapefile got above

    # after get the shapefile, they have many Jagged polygons edges. and hard to edit.
    # so, easy way is to manually draw of polygon indicate the valid regon of image in Houson, then use it to crop surface water.

    valid_shp = os.path.expanduser('~/Data/flooding_area/Houston/extent_image_valid/houston_valid_image_extent.shp')

    # global surface water
    surface_water=os.path.join(sur_water_dir,'extent_epsg2163_houston/extent_100W_30_40N_v1_3_2020.tif')
    output_raster_nodata_mask = io_function.get_name_by_adding_tail(os.path.basename(surface_water), 'res_sub_nodata')

    resample_crop_raster_using_shp(valid_shp,surface_water,output_raster_nodata_mask)



def surface_water_Houston():

    ref_raster_up = os.path.join(data_dir,
    'Houston/Houston_SAR_GRD_FLOAT_gee/S1_Houston_prj_8bit_select/S1A_IW_GRDH_1SDV_20170829T002645_20170829T002710_018131_01E74D_3220_prj_8bit.tif')

    ref_raster_down = os.path.join(data_dir,
    'Houston/Houston_SAR_GRD_FLOAT_gee/S1_Houston_prj_8bit_select/S1A_IW_GRDH_1SDV_20170829T002620_20170829T002645_018131_01E74D_D734_prj_8bit.tif')

    # notes that: these is a tiny gap between ref_raster_up and ref_raster_down, around 30 m width,
    # end in a line in the figure of surface water.
    # think about using another way to crop surface water, get shapefile of these two, then edit the shapefile and use the shapefile for cropping

    # global surface water
    surface_water=os.path.join(sur_water_dir,'extent_epsg2163_houston/extent_100W_30_40N_v1_3_2020.tif')

    # resample and crop to the same resolution and extent
    output_raster_up = io_function.get_name_by_adding_tail(os.path.basename(surface_water),'res_sub_up')
    surface_water_crop = resample_crop_raster(ref_raster_up, surface_water,output_raster=output_raster_up)

    # mask nodata regions
    out_nodata = 128        # because 255 is used to indicate ocean, so set 128 as nodata
    out_nodata_raster_up = io_function.get_name_by_adding_tail(output_raster_up,'nodataMask')
    mask_nodata_regions_surface_water(ref_raster_up, surface_water_crop, out_nodata_raster_up, ref_nodata=0, out_nodata=out_nodata)

    # resample and crop to the same resolution and extent
    output_raster_down = io_function.get_name_by_adding_tail(os.path.basename(surface_water),'res_sub_down')
    surface_water_crop = resample_crop_raster(ref_raster_down, surface_water,output_raster=output_raster_down)

    # mask nodata regions
    out_nodata_raster_down = io_function.get_name_by_adding_tail(output_raster_down,'nodataMask')
    mask_nodata_regions_surface_water(ref_raster_down, surface_water_crop, out_nodata_raster_down, ref_nodata=0, out_nodata=out_nodata)

    # merge these two?
    output_raster_nodata_mask = io_function.get_name_by_adding_tail(os.path.basename(surface_water), 'res_sub_nodata')
    command_str = 'gdal_merge.py -o %s -n %d -a_nodata %d -co compress=lzw %s %s'%(output_raster_nodata_mask,out_nodata,out_nodata,
                                                                    out_nodata_raster_up,out_nodata_raster_down)
    res = os.system(command_str)
    if res != 0:
        sys.exit(1)

    # remove the nodata, allow QGIS to custermize
    # raster_io.remove_nodata_from_raster_metadata(output_raster_nodata_mask)

    return True

def main():
    # surface_water_Houston()

    surface_water_Houston_use_shapefile()

    pass


if __name__ == '__main__':
    main()
    pass