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
import math

from rasterio.coords import BoundingBox
from rasterio.mask import mask
from rasterio.features import rasterize
from rasterio.features import shapes
from rasterio.transform import from_origin

import skimage.measure
import time
#Color interpretation https://rasterio.readthedocs.io/en/latest/topics/color.html
from rasterio.enums import ColorInterp

import basic_src.basic as basic
import basic_src.io_function as io_function

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

def get_projection(file_path, format=None):
    # https://rasterio.readthedocs.io/en/latest/api/rasterio.crs.html
    # convert the different type, to epsg, proj4, and wkt
    with rasterio.open(file_path) as src:
        if format is not None:
            if format == 'proj4':
                return src.crs.to_proj4() # string like '+init=epsg:32608', differnt from GDAL output
            elif format == 'wkt':
                return src.crs.to_wkt()     # string,  # its OGC WKT representation
            elif format == 'epsg':
                return src.crs.to_epsg()    # to epsg code, iint
            else:
                raise ValueError('Unknown format: %s'%str(format))
        return src.crs

def get_xres_yres_file(file_path):
    with rasterio.open(file_path) as src:
        xres, yres  = src.res       # Returns the (width, height) of pixels in the units of its coordinate reference system.
        return xres, yres

def get_height_width_bandnum_dtype(file_path):
    with rasterio.open(file_path) as src:
        return src.height, src.width, src.count, src.dtypes[0]

def get_transform_from_file(file_path):
    with rasterio.open(file_path) as src:
        return src.transform

def get_nodata(file_path):
    with rasterio.open(file_path) as src:
        return src.nodata

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

def get_valid_pixel_count(image_path,nodata_user=None):
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
        # nodata = src.nodata
        nodata = src.nodata if src.nodata is not None else nodata_user
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
        src.close()     # when call this function many time (> 10), the script frozen, try to manually close it.
                        # on uist, has this problem, but on tesia, don't have this problem # 2021-3-10 hlc
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

def get_valid_pixel_percentage(image_path,total_pixel_num=None, progress=None, nodata=None):
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
    valid_pixel_count, total_count = get_valid_pixel_count(image_path,nodata_user=nodata)
    if total_pixel_num is None:
        total_pixel_num =total_count

    valid_per = 100.0 * valid_pixel_count / total_pixel_num
    if progress is not None:
        print(progress, 'Done')
    return valid_per

def get_valid_percent_shannon_entropy(image_path,log_base=10,nodata_input=0, b_verbose=False):
    oneband_data, nodata = read_raster_one_band_np(image_path, band=1)
    if nodata is None:
        # raise ValueError('nodata is not set in %s, cannot tell valid pixel'%image_path)
        if b_verbose:
            print('warning, nodata is not set in %s, will use %s'%(image_path, str(nodata_input)))
        nodata = nodata_input

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


def get_location_value_attribute(raster_path,x,y,is_pixel_xy=False, attribute_name=None, band_idx=None,
                                 b_verbose = False):
    """
    get the image value of given location(x,y) in bandindex
    Args:
        imagepath:the image path which the information query
        x:x value
        y:y value
        is_pixel_xy: True for pixel coordinate
        attribute_name: the name of attribute want to get
        bandindex:the band_idx of band want to query, if it's None, then get values for all band

    Returns:the certain value (float) of given location, also attribute of band in band_idx
    """
    att_values = None
    with rasterio.open(raster_path) as src:
        indexes = src.indexes
        if band_idx is None:
            band_idx = indexes

        if is_pixel_xy is True:
            x, y = pixel_xy_to_geo_xy(x,y,src.transform)
            if b_verbose:
                print('getting values at location for %d bands: '%len(band_idx), x,y)

        # if set masked=True, it will return a mask array, to check if a point outside the raster.
        values = [item for item in src.sample([(x,y)], indexes=band_idx)]
        if attribute_name is not None:
            att_values = [src.tags(idx)[attribute_name] for idx in band_idx ]
        return values[0].tolist(), att_values



def set_nodata_to_raster_metadata(raster_path, nodata):
    # modifiy the nodata value in the metadata
    cmd_str = 'gdal_edit.py -a_nodata %s  %s' % (str(nodata), raster_path)
    print(cmd_str)
    res = os.system(cmd_str)
    if res == 0:
        return True
    else:
        return False

def remove_nodata_from_raster_metadata(raster_path):
    # modifiy the nodata value in the metadata
    cmd_str = 'gdal_edit.py -unsetnodata %s' % ( raster_path)
    print(cmd_str)
    res = os.system(cmd_str)
    if res == 0:
        return True
    else:
        return False


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
                                   compress=None, tiled=None, bigtiff=None,boundary=None,verbose=True ):
    '''
    save a numpy to file, the numpy has the same projection and extent with ref_raster
    Args:
        numpy_array:
        save_path:
        ref_raster:
        format:
        boundary: (xoff,yoff ,xsize, ysize)

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

    if verbose:
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

        if boundary is not None:
            if boundary[2] != width or boundary[3] != height:
                raise ValueError('boundary (%s) is not consistent with width (%d) and height (%d)'%(str(boundary),width,height))
            window = boundary_to_window(boundary)
            new_transform = src.window_transform(window)
            out_meta.update(transform=new_transform)

        colorinterp = [src.colorinterp[idx] for idx in range(src.count)]
        # print(colorinterp)

        with rasterio.open(save_path, "w", **out_meta) as dest:
            dest.write(numpy_array)
            # Get/set raster band color interpretation: https://github.com/mapbox/rasterio/issues/100
            if src.count == band_count:
                dest.colorinterp = colorinterp
            else:
                dest.colorinterp = [ColorInterp.undefined] * band_count

    if verbose:
        print('save to %s'%save_path)

    return True


def save_numpy_array_to_rasterfile_noCRS(numpy_array, save_path, format='GTiff',verbose=True):
    # save a numpy to file, the numpy has the same projection and extent with ref_raster, without map projection

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

    if verbose:
        print('dtype:', dt.name)
        print(numpy_array.dtype)
        print('band_count,height,width',band_count,height,width)
        # print('saved numpy_array.shape',numpy_array.shape)

    transform = from_origin(0, 0, 1, 1)  # Origin at (0, 0), pixel size (1x1)
    # Save the array as a TIFF file
    with rasterio.open(
            save_path,
            'w',
            driver=format,
            height=numpy_array.shape[1],  # Height (rows)
            width=numpy_array.shape[2],  # Width (columns)
            count=numpy_array.shape[0],  # Number of bands (3 for RGB)
            dtype=numpy_array.dtype,  # Data type of the array
            transform=transform  # Basic transform (no projection)
    ) as dst:
        dst.write(numpy_array)  # Write all bands

    if verbose:
        print('save to %s'%save_path)

    return True



def image_numpy_allBands_to_8bit_hist(img_np_allbands, min_max_values=None, per_min=0.01, per_max=0.99, bin_count = 10000, src_nodata=None, dst_nodata=None):

    input_ndim = img_np_allbands.ndim
    if input_ndim == 3:
        band_count, height, width = img_np_allbands.shape
    else:
        # add one dimension
        band_count = 1
        img_np_allbands = np.expand_dims(img_np_allbands, axis=0)

    if min_max_values is not None:
        # if we input multiple scales, it should has the same size the band count
        if len(min_max_values) > 1 and len(min_max_values) != band_count:
            raise ValueError('The number of min_max_value is not the same with band account')
        # if only input one scale, then duplicate for multiple band account.
        if len(min_max_values) == 1 and len(min_max_values) != band_count:
            min_max_values = min_max_values * band_count

    # get min, max
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
        if found_min == found_max:
            print('warning, found_min == find_max, set the output as nodata or found_min')
            new_img_np[band, :] = dst_nodata if dst_nodata is not None else found_min
        else:
            new_img_np[band,:] = image_numpy_to_8bit(img_oneband, found_max, found_min, src_nodata=src_nodata, dst_nodata=dst_nodata)

    if input_ndim == 3:
        return new_img_np
    else:
        # remove the add dimension
        return np.squeeze(new_img_np)


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
        if isinstance(src_nodata,float):
            nodata_loc = np.where(np.abs(img_np_allbands - src_nodata) < 0.00001)
        else:
            nodata_loc = np.where(img_np_allbands==src_nodata)
    band_count, height, width = img_np_allbands.shape
    # print(band_count, height, width)
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


def pixel_xy_to_geo_xy(x0,y0, transform):
    # pixel to geo XY
    # transform is from rasterio. (not GDAL)
    # https://rasterio.readthedocs.io/en/latest/topics/georeferencing.html
    x0_geo = transform[0] * x0 + transform[1] * y0 + transform[2]
    y0_geo = transform[3] * x0 + transform[4] * y0 + transform[5]
    return x0_geo, y0_geo

def pixel_xy_to_geo_xy_list(row_list, col_list, transform, offset='center'):
    # https://rasterio.readthedocs.io/en/latest/api/rasterio.transform.html#rasterio.transform.xy
    xs, ys = rasterio.transform.xy(transform,row_list,col_list, offset=offset)
    return xs, ys

def geo_xy_to_pixel_xy(x_list, y_list, transform, op=round, precision=None):

    # rows (list of ints) – list of row indices
    # cols (list of ints) – list of column indices
    rows, cols  = rasterio.transform.rowcol(transform, x_list, y_list, op=op, precision = precision)
    return cols, rows

def burn_polygon_to_raster_oneband(raster_path, polygon_shp, burn_value):
    # burn values to the extent of polygon to a raster
    # it will updated the raster

    # check projection

    shp_name = os.path.splitext(os.path.basename(polygon_shp))[0]

    # -at all touch
    cmd_str = 'gdal_rasterize -b 1 -at -burn %s -l %s '%(str(burn_value),shp_name)
    cmd_str += polygon_shp + ' ' + raster_path
    print(cmd_str)
    res = os.system(cmd_str)
    if res == 0:
        return raster_path
    else:
        return False


def burn_polygons_to_a_existing_raster(input_raster,polygons,burn_value=0):
    # burn polgyons into a existing raster
    mask_values = [1]*len(polygons)
    with rasterio.open(input_raster) as src:
        transform = src.transform
        image_data = src.read(src.indexes)  # read all bands

        burn_out = np.zeros((src.height, src.width), dtype = np.uint8)
        # rasterize the shapes
        burn_shapes = [(item_shape, item_int) for (item_shape, item_int) in
                       zip(polygons, mask_values)]
        # get the mask
        out_label = rasterize(burn_shapes, out=burn_out, transform=transform,
                              fill=0, all_touched=False, dtype=rasterio.uint8)
        # modify the image data
        loc = np.where(out_label == 1)
        image_data[:, loc[0], loc[1]] = burn_value
        kwargs = src.meta

    with rasterio.open(input_raster, 'w', **kwargs) as dst:
        dst.write(image_data)
    print('burn %s into %s'%(str(burn_value), input_raster))


def burn_polygons_to_a_raster(ref_raster, polygons, burn_values, save_path, date_type='uint8',
                              xres=None,yres=None, extent=None, ref_prj=None, nodata=None):
    # if save_path is None, it will return the array, not saving to disk
    # burn polygons to a new raster
    # if ref_raster is None, we must set xres and yres, and extent (read from polygons) and ref_prj (from shapefile)
    # extent: (minx, miny, maxx, maxy)

    if save_path is not None and os.path.isfile(save_path):
        print('%s exist, skip burn_polygons_to_a_raster'%save_path)
        return save_path

    if isinstance(burn_values,int) or isinstance(burn_values,float):
        values = [burn_values]*len(polygons)
    elif isinstance(burn_values,list):
        values = burn_values
        if len(burn_values) != len(polygons):
            raise ValueError('polygons and burn_values do not have the same size')
    else:
        raise ValueError('unkonw type of burn_values')


    if date_type=='uint8':
        save_dtype = rasterio.uint8
        np_dtype = np.uint8
    elif date_type=='uint16':
        save_dtype = rasterio.uint16
        np_dtype = np.uint16
    elif date_type == 'int32':
        save_dtype = rasterio.int32
        np_dtype = np.int32
    else:
        raise ValueError('not yet support')

    if ref_raster is None:
        # exent (minx, miny, maxx, maxy)
        height, width = math.ceil((extent[3]-extent[1])/yres), math.ceil((extent[2]-extent[0])/xres)
        burn_out = np.zeros((height, width), dtype=np_dtype)
        if nodata is not None:
            burn_out[:] = nodata
        # rasterize the shapes
        burn_shapes = [(item_shape, item_int) for (item_shape, item_int) in
                       zip(polygons, values)]
        ## new_transform = (burn_boxes[0], resX, 0, burn_boxes[3], 0, -resY )  # (X_min, resX, 0, Y_max, 0, -resY)  # GDAL-style transforms, have been deprecated after raster 1.0
        # affine.Affine() vs. GDAL-style geotransforms: https://rasterio.readthedocs.io/en/stable/topics/migrating-to-v1.html
        transform = (xres ,0, extent[0] , 0, -yres, extent[3])  # (resX, 0, X_min, 0, -resY, Y_max)
        out_label = rasterize(burn_shapes, out=burn_out, transform=transform,
                              fill=0, all_touched=False, dtype=save_dtype)

        if save_path is None:
            return out_label

        with rasterio.open(save_path, 'w', driver='GTiff',
                            height=height,
                            width=width,
                            count=1,
                            dtype=save_dtype,
                            crs=ref_prj,
                            transform=transform,
                            nodata=nodata) as dst:
            dst.write_band(1, out_label.astype(save_dtype))


    else:
        with rasterio.open(ref_raster) as src:
            transform = src.transform
            burn_out = np.zeros((src.height, src.width),dtype=np_dtype)
            if nodata is not None:
                burn_out[:] = nodata
            # rasterize the shapes
            burn_shapes = [(item_shape, item_int) for (item_shape, item_int) in
                           zip(polygons, values)]
            #
            out_label = rasterize(burn_shapes, out=burn_out, transform=transform,
                                  fill=0, all_touched=False, dtype=save_dtype)
            if save_path is None:
                return out_label

            # test: save it to disk
            kwargs = src.meta
            kwargs.update(
                dtype=save_dtype,
                count=1,
                nodata=nodata)

            # # remove nodta in the output
            # if 'nodata' in kwargs.keys():
            #     del kwargs['nodata']

            with rasterio.open(save_path, 'w', **kwargs) as dst:
                dst.write_band(1, out_label.astype(save_dtype))


def raster2shapefile(in_raster, out_shp=None, driver='ESRI Shapefile', nodata=None,connect8=True):
    # convert raster to shapefile, similar to: vector_gpd.raster2shapefile but using rasterio
    import fiona

    if out_shp is None:
        out_shp = os.path.splitext(in_raster)[0] + '.shp'
    if os.path.isfile(out_shp):
        print('%s exists, skip'%out_shp)
        return out_shp

    with rasterio.open(in_raster) as src:
        image = src.read(1)

    if nodata is not None:
        mask = image != nodata
    else:
        mask = None

    connet=4
    if connect8:
        connet=8

    results = ( {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) in enumerate(shapes(image, mask=mask, connectivity=connet, transform=src.transform)))

    # for i, (s, v) in enumerate(shapes(image, mask=mask,connectivity=8, transform=src.transform)):
    #     if i%100==0 and v==255:
    #         print(i,s,v)

    # print(results)
    # print(src.crs.to_wkt() )

    with fiona.open(
            out_shp, 'w',
            driver=driver,
            crs=src.crs.to_wkt(),
            schema={'properties': [('raster_val', 'int')],
                    'geometry': 'Polygon'}) as dst:
        dst.writerecords(results)
    return out_shp


def numpy_array_to_shape(image_2d, ref_raster, boundary=None, nodata=None, connect8=True):
    '''
    convert to mask (image_2d) to a polygon (in the shapely format)
    :param image_2d:
    :param boundary: (xoff,yoff ,xsize, ysize)
    :param ref_raster:
    :param nodata:
    :return: a list of geometries (geojson-like, use json_geometry_to_polygons to geometry), a list of raster values
    '''
    # import pprint
    if image_2d.ndim != 2:
        raise ValueError('Only support 2D image array')

    connet = 4 if connect8 is False else 8

    geometry_list = []  # geojson-like
    raster_values = []
    with rasterio.open(ref_raster) as src:
        mask = None if nodata is None else image_2d != nodata
        if boundary is not None:
            window = boundary_to_window(boundary)
            transform = src.window_transform(window)
        else:
            transform = src.transform

        out = shapes(image_2d, mask=mask, connectivity=connet, transform=transform)
        for geo_s, raster_v in out:
            geometry_list.append(geo_s)
            raster_values.append(raster_v)

        return geometry_list, raster_values


def read_colormaps_band1(raster_path):
    '''read color map for the first band'''
    # https://rasterio.readthedocs.io/en/latest/topics/color.html
    with rasterio.open(raster_path) as src:
        band_count = src.count
        if band_count > 1:
            print('warning, there are more than one band, but only read colormap for the first band')
        # print(src.colorinterp)
        colormap = None
        try:
            colormap = src.colormap(1)
        except:
            print('get colormap failed, could be: NULL color table')

        if colormap is not None:
            print('colormap for band 1 is:')
            print(colormap)

        return colormap


def write_colormaps(raster_path, color_map_dict):
    '''write or update color map for the first band
    raster_path: raster image path
    color_map_dict: a dict, like:
        color_map_dict = {0: (230,230,230,255),
                 1:(31,120,180, 255), # light blue for water
                 128:(255,255,255,255)} # nodata
    '''

    # update the raster file
    with rasterio.open(raster_path,mode='r+') as src:

        band_count = src.count
        if band_count > 1:
            print('warning, there are more than one band, but only update colormap for the first band')

        src.write_colormap(
            1, color_map_dict)

        print('Write color map to %s'%raster_path)

        # read and check
        # cmap = src.colormap(1)
        # # True
        # assert cmap[0] == (255, 0, 0, 255)
        # # True
        # assert cmap[1] == (0, 0, 255, 255)

    return True

def read_metadata_bands(raster_path, attribute_name, band_idx=None ):
    with rasterio.open(raster_path) as src:
        indexes = src.indexes
        if band_idx is None:
            band_idx = indexes

        if attribute_name is not None:
            att_values = [src.tags(idx)[attribute_name] for idx in band_idx ]
        return att_values

def add_metadata_to_bands(raster_path, meta_name, meta_values):

    # update the raster file
    with rasterio.open(raster_path,mode='r+') as src:

        band_count = src.count
        if band_count != len(meta_values):
            raise ValueError('The number of band account (%d) and values (%d) does not match'%(band_count, len(meta_values)))

        for idx in range(band_count):
            src.update_tags(idx+1, **{meta_name:meta_values[idx]})

def set_band_description(raster_path, band_description_list, band_idx=None):
    # set band name (band description) for each band
    # band_description_list (a list) for the band descriptions
    # band_idx, the band index need to set, also a list (first band is 1, not 0)

    # update the raster file
    with rasterio.open(raster_path,mode='r+') as src:
        indexes = src.indexes
        if band_idx is None:
            band_idx = indexes

        if len(band_description_list) != len(band_idx):
            raise ValueError('The count of band descriptions (%d) and indexes (%d) is different'%(len(band_description_list), len(band_idx)))

        for ii, b_idx in enumerate(band_idx):
            src.set_band_description(b_idx, band_description_list[ii])


def trim_nodata_region(img_path, save_path,nodata=0, tmp_dir='./'):
    # crop the nodata region of a raster
    ## trim nodata region
    # calculate the valid region
    mask_tif = os.path.join(tmp_dir, io_function.get_name_no_ext(img_path) +'_mask.tif')
    cmd_str = 'gdal_calc.py --type=Byte --NoDataValue=%s  --calc="A!=%s" -A %s --outfile=%s'%(str(nodata),str(nodata),img_path,mask_tif)
    basic.os_system_exit_code(cmd_str)
    io_function.is_file_exist(mask_tif)

    # polygonize
    outline_shp = os.path.join(tmp_dir, io_function.get_name_no_ext(img_path) +'_outline.shp')
    cmd_str = 'gdal_polygonize.py -8  %s  -b 1 -f "ESRI Shapefile" %s'%(mask_tif,outline_shp)
    basic.os_system_exit_code(cmd_str)
    io_function.is_file_exist(mask_tif)

    # crop, and translate to Geotiff
    cmd_str = 'gdalwarp -of GTiff -co compress=lzw -co tiled=yes -co bigtiff=if_safer -cutline %s -crop_to_cutline '%outline_shp + img_path + ' ' + save_path
    basic.os_system_exit_code(cmd_str)

    ## a safe way to crop is use the bounding box, because something a polygon in outline.shp may be invalid.
    ##  but here we need to call function in vector_gpd   noted by hlc, 2023, May 22.
    # bounding_boxes = vector_gpd.get_vector_file_bounding_box(outline_shp)  # minx, miny, maxx, maxy

    # if np.any(np.isnan(bounding_boxes) == True):  # nan
    #   handle this

    # box_str = ' '.join([str(item) for item in bounding_boxes])
    # cmd_str = 'gdalwarp -of GTiff -co compress=lzw -co tiled=yes -co bigtiff=if_safer -te %s '%box_str + img_path + ' ' + save_path
    # basic.os_system_exit_code(cmd_str)

    return save_path

def convert_images_to_rgb_8bit_np(img_path,save_path=None,rgb_bands=[1,2,3], sr_min=0, sr_max=2000, nodata=0, format='GTiff',verbose=False):
    ### convert an image to 8bit and keep the RGB bands, using Numpy

    if save_path is None:
        save_path = io_function.get_name_by_adding_tail(img_path,'rgb_8bit')
    if os.path.isfile(save_path):
        print(f'{save_path} already exists, skip')
        return

    img_np_allbands, src_nodata = read_raster_all_bands_np(img_path)
    img_shape = img_np_allbands.shape
    # print(img_np_allbands.shape, src_nodata)
    # if more than three bands
    if img_shape[0] > 3:
        img_np_allbands = img_np_allbands[[ item-1 for item in rgb_bands] ,:,:]
    # need to handle band number less than 3

    # print(img_np_allbands.shape, src_nodata)

    # scales = (src_min src_max dst_min dst_max)
    scales = [(sr_min, sr_max, 1, 255)]


    # print('input scale (src_min src_max dst_min dst_max): ' + str(scales))
    img_array_8bit = image_numpy_allBands_to_8bit(img_np_allbands, scales, src_nodata=src_nodata,
                                                            dst_nodata=nodata)

    # min_percent = 0.02
    # max_percent = 0.98
    # min_max_value = None
    # hist_bin_count = 1000
    # img_array_8bit = image_numpy_allBands_to_8bit_hist(img_np_allbands, min_max_value,
    #                                                              per_min=min_percent,
    #                                                              per_max=max_percent, bin_count=hist_bin_count,
    #                                                              src_nodata=src_nodata,
    #                                                              dst_nodata=nodata)

    save_numpy_array_to_rasterfile(img_array_8bit, save_path, img_path, format=format,
                                                    nodata=nodata, compress='lzw', tiled='yes', bigtiff='if_safer', verbose=verbose)

    return save_path


def convert_images_to_rgb_8bit_gdal(img_path,save_path=None,rgb_bands=[1,2,3], sr_min=0, sr_max=2000, nodata=None, format='GTiff'):
    ### convert an image to 8bit and keep the RGB bands, using gdal_translate, for large images
    # filename_no_ext
    if save_path is None:
        save_path = io_function.get_name_by_adding_tail(img_path,'rgb_8bit')

    if os.path.isfile(save_path):
        print(f'{save_path} already exists, skip')
        return

    # extract  rgb bands
    save_path_rgb = io_function.get_name_by_adding_tail(img_path,'rgb')
    cmd_str = f'gdal_translate -b {rgb_bands[0]} -b {rgb_bands[1]} -b {rgb_bands[2]} -of VRT {img_path} {save_path_rgb}'
    basic.os_system_exit_code(cmd_str)


    # to 8bit
    # save_8bit = io_function.get_name_by_adding_tail(save_path_rgb,'8bit')
    # use fix min and max to make the color be consistent to sentinel-images
    src_min = sr_min
    src_max = sr_max
    dst_min = 1  # 0 is the nodata, so set as 1
    dst_max = 255
    # format: "-of VRT"  or "-of GTiff"  or "-of PNG" (for web display)
    cmd_str = f'gdal_translate -ot Byte -scale {src_min} {src_max} {dst_min} {dst_max} -of {format} {save_path_rgb} {save_path}'  # -of VRT

    basic.os_system_exit_code(cmd_str)

    # set nodata
    if nodata is not None:
        # gdal_edit.py -a_nodata 0  ${fin_output}
        cmd_str = f'gdal_edit.py -a_nodata {nodata}  {save_path}'
        basic.os_system_exit_code(cmd_str)

    io_function.delete_file_or_dir(save_path_rgb)

    return save_path




def main():
    pass


if __name__=='__main__':
    main()
