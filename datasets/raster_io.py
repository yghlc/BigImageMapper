#!/usr/bin/env python
# Filename: raster_io 
"""
introduction:  Based on rasterio, to read and write raster data

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 03 January, 2021
"""

import os, sys
from optparse import OptionParser
import rasterio
import numpy as np

from rasterio.coords import BoundingBox
from rasterio.mask import mask

import skimage.measure
import time
#Color interpretation https://rasterio.readthedocs.io/en/latest/topics/color.html
from rasterio.enums import ColorInterp

def open_raster_read(raster_path):
    src = rasterio.open(raster_path)
    return src

def get_width_heigth_bandnum(opened_src):
    return opened_src.height,  opened_src.width,  opened_src.count

# def get_xres_yres(opened_src):
#     return opened_src.height,  opened_src.width,  opened_src.count

def get_driver_format(file_path):
    with rasterio.open(file_path) as src:
        return src.driver

def get_projection(file_path):
    # https://rasterio.readthedocs.io/en/latest/api/rasterio.crs.html
    # convert the different type, to epsg, proj4, and wkt
    with rasterio.open(file_path) as src:
        return src.crs

def get_xres_yres_file(file_path):
    with rasterio.open(file_path) as src:
        xres, yres  = src.res       # Returns the (width, height) of pixels in the units of its coordinate reference system.
        return xres, yres

def get_height_width_bandnum_dtype(file_path):
    with rasterio.open(file_path) as src:
        return src.height, src.width, src.count, src.dtypes[0]

def get_area_image_box(file_path):
    # get the area of an image coverage (including nodata area)
    with rasterio.open(file_path) as src:
        # the extent of the raster
        raster_bounds = src.bounds  # (left, bottom, right, top)
        height = raster_bounds.top - raster_bounds.bottom
        width = raster_bounds.right - raster_bounds.left
        return height*width

def get_image_bound_box(file_path, buffer=None):
    # get the bounding box: (left, bottom, right, top)
    with rasterio.open(file_path) as src:
        # the extent of the raster
        raster_bounds = src.bounds
        if buffer is not None:
            # Create new instance of BoundingBox(left, bottom, right, top)
            new_box_obj = BoundingBox(raster_bounds.left-buffer, raster_bounds.bottom-buffer,
                       raster_bounds.right+buffer, raster_bounds.top+ buffer)
            # print(raster_bounds, new_box_obj)
            return new_box_obj
        return raster_bounds

def get_valid_pixel_count(image_path):
    """
    get the count of valid pixels (exclude no_data pixel)
    assume that the nodata value already be set
    Args:
        image_path: path

    Returns: the count

    """

    t0 = time.time()
    band = 1
    # count the pixel block by block,  quicker than read the entire image
    # https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html?highlight=block_shapes#blocks
    valid_pixel_count = 0
    total_count = 0
    with rasterio.open(image_path) as src:
        assert len(set(src.block_shapes)) == 1   # check have identically blocked bands
        # for i, shape in enumerate(src.block_shapes,start=1):    # output shape
        #     print((i, shape))
        nodata = src.nodata
        # print(nodata)
        if nodata is None:
            raise ValueError('nodata is not set in %s, cannot tell valid pixel' % image_path)
        for ji, window in src.block_windows(band):     # 1 mean for band one
            # print((ji, window))
            band_block_data = src.read(band, window=window) # it seems that src convert nodata to nan automatically
            # print(band_block_data.shape)
            # print(band_block_data)
            valid_loc = np.where(band_block_data != nodata)
            # if band_block_data.dtype == 'float32':
            # always check nan
            nan_loc = np.where(np.isnan(band_block_data))
            valid_pixel_count -=  nan_loc[0].size

            valid_pixel_count += valid_loc[0].size
            total_count += band_block_data.size
            # break
    # total_count = src.width*src.height
    # print('valid_pixel_count, total_count, time cost',valid_pixel_count, total_count,time.time() - t0)
    return valid_pixel_count, total_count

    # # read the entire image, then calculate
    # oneband_data, nodata = read_raster_one_band_np(image_path, band=1)
    # if nodata is None:
    #     raise ValueError('nodata is not set in %s, cannot tell valid pixel'%image_path)
    #
    # valid_loc = np.where(oneband_data != nodata)
    # valid_pixel_count = valid_loc[0].size
    # if oneband_data.dtype == 'float32':
    #     nan_loc = np.where(np.isnan(oneband_data))
    #     valid_pixel_count -=  nan_loc[0].size
    #
    # # return valid count and total count
    # print('valid_pixel_count, total_count, time cost', valid_pixel_count, oneband_data.size, time.time() - t0)
    # return valid_pixel_count, oneband_data.size

def get_valid_pixel_percentage(image_path,total_pixel_num=None, progress=None):
    """
    get the percentage of valid pixels (exclude no_data pixel)
    assume that the nodata value already be set
    Args:
        image_path: path
        total_pixel_num: total pixel count, for example, the image only cover a portion of the area
        progress: to show the progress when parallel call this function

    Returns: the percentage (%)

    """
    if progress is not None:
        print(progress)
    valid_pixel_count, total_count = get_valid_pixel_count(image_path)
    if total_pixel_num is None:
        total_pixel_num =total_count

    valid_per = 100.0 * valid_pixel_count / total_pixel_num
    if progress is not None:
        print(progress, 'Done')
    return valid_per

def get_valid_percent_shannon_entropy(image_path,log_base=10):
    oneband_data, nodata = read_raster_one_band_np(image_path, band=1)
    if nodata is None:
        raise ValueError('nodata is not set in %s, cannot tell valid pixel'%image_path)

    valid_loc = np.where(oneband_data != nodata)
    valid_pixel_count = valid_loc[0].size
    total_count = oneband_data.size

    valid_per = 100.0 * valid_pixel_count / total_count
    entropy = skimage.measure.shannon_entropy(oneband_data, base=log_base)

    return valid_per, entropy

def get_max_min_histogram_percent_oneband(data, bin_count, min_percent=0.01, max_percent=0.99, nodata=None,
                                          hist_range=None):
    '''
    get the max and min when cut of % top and bottom pixel values
    :param data: one band image data, 2d array.
    :param bin_count: bin_count of calculating the histogram
    :param min_percent: percent
    :param max_percent: percent
    :param nodata:
    :param hist_range: [min, max] for calculating the histogram
    :return: min, max value, histogram (hist, bin_edges)
    '''
    if data.ndim != 2:
        raise ValueError('Only accept 2d array')
    data_1d = data.flatten()
    if nodata is not None:
        data_1d = data_1d[data_1d != nodata] # remove nodata values

    data_1d = data_1d[~np.isnan(data_1d)]   # remove nan value
    hist, bin_edges = np.histogram(data_1d, bins=bin_count, density=False, range=hist_range)

    # get the min and max based on percent cut.
    if min_percent >= max_percent:
        raise ValueError('min_percent >= max_percent')
    found_min = 0
    found_max = 0

    count = hist.size
    sum = np.sum(hist)
    accumulate_sum = 0
    for ii in range(count):
        accumulate_sum += hist[ii]
        if accumulate_sum/sum >= min_percent:
            found_min = bin_edges[ii]
            break

    accumulate_sum = 0
    for ii in range(count-1,0,-1):
        # print(ii)
        accumulate_sum += hist[ii]
        if accumulate_sum / sum >= (1 - max_percent):
            found_max = bin_edges[ii]
            break

    return found_min, found_max, hist, bin_edges

def is_two_bound_disjoint(box1, box2):
    # box1 and box2: bounding box: (left, bottom, right, top)
    # Compare two bounds and determine if they are disjoint.
    # return True if bounds are disjoint, False if they are overlap
    return rasterio.coords.disjoint_bounds(box1,box2)

def is_two_image_disjoint(img1, img2, buffer=None):
    box1 = get_image_bound_box(img1, buffer=buffer)
    box2 = get_image_bound_box(img2, buffer=buffer)
    return is_two_bound_disjoint(box1,box2)


def read_oneband_image_to_1dArray(image_path,nodata=None, ignore_small=None):

    if os.path.isfile(image_path) is False:
        raise IOError("error, file not exist: " + image_path)

    with rasterio.open(image_path) as img_obj:
        # read the all bands (only have one band)
        indexes = img_obj.indexes
        if len(indexes) != 1:
            raise IOError('error, only support one band')

        data = img_obj.read(indexes)
        data_1d = data.flatten()  # convert to one 1d, row first.

        # input nodata
        if nodata is not None:
            data_1d = data_1d[data_1d != nodata]
        # the nodata in the image meta.
        if img_obj.nodata is not None:
            data_1d = data_1d[data_1d != img_obj.nodata]

        if ignore_small is not None:
            data_1d = data_1d[data_1d >= ignore_small ]

        return data_1d

def boundary_to_window(boundary):
    # boundary: (xoff,yoff ,xsize, ysize)
    # window structure; expecting ((row_start, row_stop), (col_start, col_stop))
    window = ((boundary[1],boundary[1]+boundary[3])  ,  (boundary[0],boundary[0]+boundary[2]))
    return window

def read_raster_in_polygons_mask(raster_path, polygons, nodata=None, all_touched=True, crop=True,
                                 bands = None, save_path=None):
    # using mask to get pixels in polygons
    # see more information of the parameter in the function: mask

    if isinstance(polygons, list) is False:
        polygon_list = [polygons]
    else:
        polygon_list = polygons

    with rasterio.open(raster_path) as src:
        # crop image and saved to disk
        out_image, out_transform = mask(src, polygon_list, nodata=nodata, all_touched=all_touched, crop=crop,
                                        indexes=bands)

        # print(out_image.shape)
        if out_image.ndim == 2:
            height, width = out_image.shape
            band_count = 1
        else:
            band_count, height, width = out_image.shape
        if nodata is None:  # if it None, copy from the src file
            nodata = src.nodata
        if save_path is not None:
            # save it to disk
            out_meta = src.meta.copy()
            out_meta.update({"driver": "GTiff",
                             "height": height,
                             "width": width,
                             "count": band_count,
                             "transform": out_transform,
                             "nodata": nodata})  # note that, the saved image have a small offset compared to the original ones (~0.5 pixel)
            if out_image.ndim == 2:
                out_image = out_image.reshape((1, height, width))
            with rasterio.open(save_path, "w", **out_meta) as dest:
                dest.write(out_image)

        return out_image, out_transform, nodata

def read_raster_all_bands_np(raster_path, boundary=None):
    # boundary: (xoff,yoff ,xsize, ysize)

    with rasterio.open(raster_path) as src:
        indexes = src.indexes
        
        if boundary is not None:
            data = src.read(indexes, window=boundary_to_window(boundary))
        else:
            data = src.read(indexes)   # output (band_count, height, width)

        # print(data.shape)
        # print(src.nodata)
        # if src.nodata is not None and src.dtypes[0] == 'float32':
        #     data[ data == src.nodata ] = np.nan

        return data, src.nodata

def read_raster_one_band_np(raster_path,band=1,boundary=None):
    # boundary: (xoff,yoff ,xsize, ysize)
    with rasterio.open(raster_path) as src:

        if boundary is not None:
            data = src.read(band, window=boundary_to_window(boundary))
        else:
            data = src.read(band)       # output (height, width)

        # if src.nodata is not None and src.dtypes[0] == 'float32':
        #     data[ data == src.nodata ] = np.nan
        return data, src.nodata

def save_numpy_array_to_rasterfile(numpy_array, save_path, ref_raster, format='GTiff', nodata=None,
                                   compress=None, tiled=None, bigtiff=None):
    '''
    save a numpy to file, the numpy has the same projection and extent with ref_raster
    Args:
        numpy_array:
        save_path:
        ref_raster:
        format:

    Returns:

    '''
    if numpy_array.ndim == 2:
        band_count = 1
        height,width = numpy_array.shape
        # reshape to 3 dim, to write the disk
        numpy_array = numpy_array.reshape(band_count, height, width)
    elif numpy_array.ndim == 3:
        band_count, height,width = numpy_array.shape
    else:
        raise ValueError('only accept ndim is 2 or 3')

    dt = np.dtype(numpy_array.dtype)

    print('dtype:', dt.name)
    print(numpy_array.dtype)
    print('band_count,height,width',band_count,height,width)
    # print('saved numpy_array.shape',numpy_array.shape)

    with rasterio.open(ref_raster) as src:
        # [print(src.colorinterp[idx]) for idx in range(src.count)]
        # test: save it to disk
        out_meta = src.meta.copy()
        out_meta.update({"driver": format,
                         "height": height,
                         "width": width,
                         "count":band_count,
                         "dtype": dt.name
                         })
        if nodata is not None:
            out_meta.update({"nodata": nodata})

        if compress is not None:
            out_meta.update(compress=compress)
        if tiled is not None:
            out_meta.update(tiled=tiled)
        if bigtiff is not None:
            out_meta.update(bigtiff=bigtiff)

        colorinterp = [src.colorinterp[idx] for idx in range(src.count)]
        # print(colorinterp)

        with rasterio.open(save_path, "w", **out_meta) as dest:
            dest.write(numpy_array)
            # Get/set raster band color interpretation: https://github.com/mapbox/rasterio/issues/100
            if src.count == band_count:
                dest.colorinterp = colorinterp
            else:
                dest.colorinterp = [ColorInterp.undefined] * band_count

    print('save to %s'%save_path)

    return True

def image_numpy_allBands_to_8bit_hist(img_np_allbands, min_max_values=None, per_min=0.01, per_max=0.99, src_nodata=None, dst_nodata=None):

    band_count, height, width = img_np_allbands.shape
    if min_max_values is not None:
        # if we input multiple scales, it should has the same size the band count
        if len(min_max_values) > 1 and len(min_max_values) != band_count:
            raise ValueError('The number of min_max_value is not the same with band account')
        # if only input one scale, then duplicate for multiple band account.
        if len(min_max_values) == 1 and len(min_max_values) != band_count:
            min_max_value = min_max_values * band_count

    # get min, max
    bin_count = 500
    new_img_np = np.zeros_like(img_np_allbands, dtype=np.uint8)
    for band, img_oneband in enumerate(img_np_allbands):
        found_min, found_max, hist, bin_edges = get_max_min_histogram_percent_oneband(img_oneband, bin_count,
                                                                                                min_percent=per_min,
                                                                                                max_percent=per_max,
                                                                                                nodata=src_nodata)
        print('min and max value from histogram (percent cut):', found_min, found_max)
        if min_max_values is not None:
            if found_min < min_max_values[band][0]:
                found_min = min_max_values[band][0]
                print('reset the min value to %s' % found_min)
            if found_max > min_max_values[band][1]:
                found_max = min_max_values[band][1]
                print('reset the max value to %s' % found_max)
        new_img_np[band,:] = image_numpy_to_8bit(img_oneband, found_max, found_min, src_nodata=src_nodata, dst_nodata=dst_nodata)

    return new_img_np


def image_numpy_allBands_to_8bit(img_np_allbands, scales, src_nodata=None, dst_nodata=None):
    '''
    linear scretch and save to 8 bit.
    Args:
        img_np:
        scales: one or multiple list of (src_min src_max dst_min dst_max)
        src_nodata:
        dst_nodata:

    Returns: new numpy array

    '''
    nodata_loc = None
    if src_nodata is not None:
        nodata_loc = np.where(img_np_allbands==src_nodata)
    band_count, height, width = img_np_allbands.shape
    print(band_count, height, width)
    # if we input multiple scales, it should has the same size the band count
    if len(scales) > 1 and len(scales) != band_count:
        raise ValueError('The number of scales is not the same with band account')
    # if only input one scale, then duplicate for multiple band account.
    if len(scales)==1 and len(scales) != band_count:
        scales = scales*band_count

    new_img_np = np.zeros_like(img_np_allbands,dtype=np.uint8)

    for idx, (scale, img_oneband) in enumerate(zip(scales,img_np_allbands)):
        # print(scale)
        # print(img_oneband.shape)

        src_min = float(scale[0])
        scr_max = float(scale[1])
        dst_min = float(scale[2])
        dst_max = float(scale[3])
        img_oneband[img_oneband > scr_max] = scr_max
        img_oneband[img_oneband < src_min] = src_min

        # scale the grey values to dst_min - dst_max
        k = (dst_max - dst_min) * 1.0 / (scr_max - src_min)
        new_img_np[idx,:] = (img_oneband - src_min) * k + dst_min

    # replace nodata
    if nodata_loc is not None and nodata_loc[0].size >0:
        if dst_nodata is not None:
            new_img_np[nodata_loc] = dst_nodata
        else:
            new_img_np[nodata_loc] = src_nodata

    return new_img_np


def image_numpy_to_8bit(img_np, max_value, min_value, src_nodata=None, dst_nodata=None):
    '''
    convert float or 16 bit to 8bit,
    Args:
        img_np:  numpy array
        max_value:
        min_value:
        src_nodata:
        dst_nodata:  if output nodata is 0, then covert data to 1-255, if it's 255, then to 0-254

    Returns: new numpy array

    '''
    print('Convert to 8bit, original max, min: %.4f, %.4f'%(max_value, min_value))
    nan_loc = np.where(np.isnan(img_np))
    if nan_loc[0].size > 0:
        img_np = np.nan_to_num(img_np)

    nodata_loc = None
    if src_nodata is not None:
        nodata_loc = np.where(img_np==src_nodata)

    img_np[img_np > max_value] = max_value
    img_np[img_np < min_value] = min_value

    if dst_nodata == 0:
        n_max, n_min = 255, 1
    elif dst_nodata == 255:
        n_max, n_min = 254, 0
    else:
        n_max, n_min = 255, 0

    # scale the grey values to 0 - 255 for better display
    k = (n_max - n_min)*1.0/(max_value - min_value)
    new_img_np = (img_np - min_value) * k + n_min
    new_img_np = new_img_np.astype(np.uint8)

    # replace nan data as nodata
    if nan_loc[0].size > 0:
        if dst_nodata is not None:
            new_img_np[nan_loc] = dst_nodata
        else:
            new_img_np[nan_loc] = n_min
    # replace nodata
    if nodata_loc is not None and nodata_loc[0].size >0:
        if dst_nodata is not None:
            new_img_np[nodata_loc] = dst_nodata
        else:
            new_img_np[nodata_loc] = src_nodata

    return new_img_np

def main():
    pass


if __name__=='__main__':
    main()
