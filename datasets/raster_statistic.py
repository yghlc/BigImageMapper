#!/usr/bin/env python
# Filename: raster_statistic 
"""
introduction: conduct statistic based on vectos, similar to https://github.com/perrygeo/python-rasterstats,
#           but allow image tiles (multi-raster).

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 02 March, 2021
"""

import os,sys
import vector_gpd
from shapely.geometry import mapping # transform to GeJSON format

import raster_io
import basic_src.io_function as io_function
import basic_src.map_projection as map_projection
import basic_src.basic as basic
import numpy as np


from multiprocessing import Pool

def array_stats(in_array, stats, nodata,range=None):
    data_1d = in_array.flatten()
    data_1d = data_1d[ data_1d != nodata]
    data_1d = data_1d[~np.isnan(data_1d)]  # remove nan value

    if range is not None:
        lower = range[0]
        upper = range[1]
        if lower is None:
            data_1d = data_1d[data_1d <= upper]
        elif upper is None:
            data_1d = data_1d[data_1d >= lower]
        else:
            data_1d = data_1d[np.logical_and( data_1d >= lower, data_1d <= upper ) ]

    # https://numpy.org/doc/stable/reference/routines.statistics.html
    out_value_dict = {}
    if data_1d.size == 0:
        for item in stats:
            if item == 'count':
                out_value_dict[item] = data_1d.size
                continue
            out_value_dict[item] = None

        return out_value_dict

    for item in stats:
        if item == 'mean':
            value = np.mean(data_1d)
        elif item == 'max':
            value = np.max(data_1d)
        elif item == 'min':
            value = np.min(data_1d)
        elif item == 'median':
            value = np.median(data_1d)
        elif item == 'count':
            value = data_1d.size
        elif item =='std':
            value = np.std(data_1d)
        else:
            raise ValueError('unsupported stats: %s'%item)
        out_value_dict[item] = value

    return out_value_dict


def zonal_stats_one_polygon(idx, polygon, image_tiles, img_tile_polygons, stats, nodata=None,range=None,
                            band = 1,all_touched=True, tile_min_overlap=None):

    overlap_index = vector_gpd.get_poly_index_within_extent(img_tile_polygons, polygon,min_overlap_area=tile_min_overlap)
    image_list = [image_tiles[item] for item in overlap_index]

    if len(image_list) == 1:
        out_image, out_tran,nodata = raster_io.read_raster_in_polygons_mask(image_list[0], polygon, nodata=nodata,
                                                                     all_touched=all_touched,bands=band)
    elif len(image_list) > 1:
        # for the case it overlap more than one raster, need to produce a mosaic
        tmp_saved_files = []
        for k_img, image_path in enumerate(image_list):

            # print(image_path)
            tmp_save_path = os.path.splitext(os.path.basename(image_path))[0] + '_subset_poly%d'%idx +'.tif'
            _, _,nodata = raster_io.read_raster_in_polygons_mask(image_path, polygon,all_touched=all_touched,nodata=nodata,
                                                          bands=band, save_path=tmp_save_path)
            tmp_saved_files.append(tmp_save_path)

        # mosaic files in tmp_saved_files
        save_path = 'raster_for_poly%d.tif'%idx
        mosaic_args_list = ['gdal_merge.py', '-o', save_path,'-n',str(nodata),'-a_nodata',str(nodata)]
        mosaic_args_list.extend(tmp_saved_files)
        if basic.exec_command_args_list_one_file(mosaic_args_list,save_path) is False:
            raise IOError('error, obtain a mosaic (%s) failed'%save_path)

        # read the raster
        out_image, out_nodata = raster_io.read_raster_one_band_np(save_path,band=band)
        # remove temporal raster
        tmp_saved_files.append(save_path)
        for item in tmp_saved_files:
            io_function.delete_file_or_dir(item)

    else:
        basic.outputlogMessage('warning, cannot find raster for %d (start=0) polygon'%idx)
        # return None           # dont return None, we cause error, let array_stats handle the empty array
        out_image = np.array([])

    # do calculation
    return array_stats(out_image, stats, nodata,range=range)

def zonal_stats_multiRasters(in_shp, raster_file_or_files, tile_min_overlap=None, nodata=None, band = 1, stats = None, prefix='',
                             range=None,buffer=None, all_touched=True, process_num=1):
    '''
    zonal statistic based on vectors, along multiple rasters (image tiles)
    Args:
        in_shp: input vector file
        raster_file_or_files: a raster file or multiple rasters
        nodata:
        band: band
        stats: like [mean, std, max, min]
        range: interested values [min, max], None means infinity
        buffer: expand polygon with buffer (meter) before the statistic
        all_touched:
        process_num: process number for calculation

    Returns:

    '''
    io_function.is_file_exist(in_shp)
    if stats is None:
        basic.outputlogMessage('warning, No input stats, set to ["mean"])')
        stats = ['mean']
    stats_backup = stats.copy()
    if 'area' in stats:
        stats.remove('area')
        if 'count' not in stats:
            stats.append('count')

    if isinstance(raster_file_or_files,str):
        io_function.is_file_exist(raster_file_or_files)
        image_tiles = [raster_file_or_files]
    elif isinstance(raster_file_or_files,list):
        image_tiles = raster_file_or_files
    else:
        raise ValueError('unsupport type for %s'%str(raster_file_or_files))

    # check projection (assume we have the same projection), check them outside this function

    # get image box
    img_tile_boxes =  [raster_io.get_image_bound_box(tile) for tile in image_tiles]
    img_tile_polygons = [vector_gpd.convert_image_bound_to_shapely_polygon(box) for box in img_tile_boxes]
    polygons = vector_gpd.read_polygons_gpd(in_shp)
    if len(polygons) < 1:
        basic.outputlogMessage('No polygons in %s'%in_shp)
        return False
    # polygons_json = [mapping(item) for item in polygons]  # no need when use new verion of rasterio
    if buffer is not None:
        polygons = [ poly.buffer(buffer) for poly in polygons]

    # process polygons one by one polygons and the corresponding image tiles (parallel and save memory)
    # also to avoid error: daemonic processes are not allowed to have children
    if process_num == 1:
        stats_res_list = []
        for idx, polygon in enumerate(polygons):
            out_stats = zonal_stats_one_polygon(idx, polygon, image_tiles, img_tile_polygons, stats, nodata=nodata, range=range,
                                    band=band, all_touched=all_touched)
            stats_res_list.append(out_stats)

    elif process_num > 1:
        threadpool = Pool(process_num)
        para_list = [ (idx, polygon, image_tiles, img_tile_polygons, stats, nodata, range,band, all_touched)
                      for idx, polygon in enumerate(polygons)]
        stats_res_list = threadpool.starmap(zonal_stats_one_polygon,para_list)
    else:
        raise ValueError('Wrong process number: %s '%str(process_num))



    # save to shapefile
    add_attributes = {}
    new_key_list  = [ prefix + '_' + key for key in stats_res_list[0].keys()]
    for new_ley in new_key_list:
        add_attributes[new_ley] = []
    for stats_result in stats_res_list:
        for key in stats_result.keys():
            add_attributes[prefix + '_' + key].append(stats_result[key])

    if 'area' in stats_backup:
       dx, dy = raster_io.get_xres_yres_file(image_tiles[0])
       add_attributes[prefix + '_' + 'area'] = [ count*dx*dy for count in add_attributes[prefix + '_' + 'count'] ]

       if 'count' not in stats_backup:
            del add_attributes[prefix + '_' + 'count']

    vector_gpd.add_attributes_to_shp(in_shp,add_attributes)

    pass


def test_zonal_stats_multiRasters():

    shp = os.path.expanduser('~/Data/Arctic/canada_arctic/Willow_River/Willow_River_Thaw_Slumps.shp')
    # save_shp = os.path.basename(io_function.get_name_by_adding_tail(shp,'raster_stats'))


    # a single DEM
    # dem_file_dir = os.path.expanduser('~/Data/Arctic/canada_arctic/DEM/WR_dem_ArcticDEM_mosaic')
    # dem_path = os.path.join(dem_file_dir,'WR_extent_2m_v3.0_ArcticTileDEM_sub_1_prj.tif')

    # dem patches
    dem_file_dir = os.path.expanduser('~/Data/Arctic/canada_arctic/DEM/WR_dem_ArcticDEM_mosaic/dem_patches')
    dem_list = io_function.get_file_list_by_ext('.tif',dem_file_dir,bsub_folder=False)
    save_shp = os.path.basename(io_function.get_name_by_adding_tail(shp, 'multi_raster_stats'))

    io_function.copy_shape_file(shp, save_shp)
    zonal_stats_multiRasters(save_shp, dem_list, nodata=None, band=1, stats=None, prefix='dem',
                             range=None, all_touched=True, process_num=4)



def main():
    test_zonal_stats_multiRasters()
    pass

if __name__=='__main__':
    basic.setlogfile('raster_statistic.log')
    main()
