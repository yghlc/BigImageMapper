#!/usr/bin/env python
# Filename: map_projection.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 05 May, 2016
"""

# support for python3: import the script in the same dir. hlc 2019-Jan-15
import sys,os
py_dir=os.path.dirname(os.path.realpath(__file__))
# print(py_dir)
sys.path.append(py_dir)

import sys,basic
from RSImage import RSImageclass
import math


import io_function

def wkt_to_proj4(wkt):
    srs = osr.SpatialReference()
    srs.importFromWkt(wkt)
    proj4 = srs.ExportToProj4()
    if proj4 is False:
        basic.outputlogMessage('convert wkt to proj4 failed')
        return False
    return proj4

def proj4_to_wkt(proj4):
    srs = osr.SpatialReference()
    srs.ImportFromProj4(proj4)
    wkt = srs.ExportToWkt()
    if wkt is False:
        basic.outputlogMessage('convert wkt to proj4 failed')
        return False
    return wkt

def convert_pixel_xy_to_lat_lon(pixel_x,pixel_y,ref_image):
    '''
    get lat, lon (WGS 84) of a pixel point
    Args:
        pixel_x:
        pixel_y:
        ref_image:

    Returns:

    '''
    x_map, y_map = convert_pixel_xy_to_map_coordinate(pixel_x,pixel_y,ref_image)
    epsg_info = get_raster_or_vector_srs_info(ref_image,'epsg')

    epsg_int = int(epsg_info.split(':')[1])

    if epsg_info=='EPSG:4326': # alreay on lat, lon
        return x_map, y_map
    else:
        # to list for the input
        x=[x_map]
        y=[y_map]
        if convert_points_coordinate_epsg(x,y,epsg_int,4326): # to 'EPSG:4326'
            return x[0], y[0]
        else:
            raise ValueError('error in convert coordinates')

def convert_pixel_xy_to_map_coordinate(pixel_x,pixel_y,ref_image):
    """
    get the map x,y of a pixel point
    Args:
        pixel_x: pixel, column index
        pixel_y: pixel, row index
        ref_image: the georeference image

    Returns: x_map, y_map

    """
    img_obj = RSImageclass()
    if img_obj.open(ref_image) is False:
        raise IOError('Open %s failed'%ref_image)

    # pixel to map coordiante
    # https://www.gdal.org/classGDALDataset.html#a5101119705f5fa2bc1344ab26f66fd1d
    # Xp = padfTransform[0] + P*padfTransform[1] + L*padfTransform[2];
    # Yp = padfTransform[3] + P*padfTransform[4] + L*padfTransform[5];
    padfTransform = img_obj.GetGeoTransform()

    x_map = padfTransform[0] + pixel_x*padfTransform[1] + pixel_y*padfTransform[2]
    y_map = padfTransform[3] + pixel_x*padfTransform[4] + pixel_y*padfTransform[5]
    return x_map, y_map



def convert_points_SpatialRef(input_x,input_y,inSpatialRef,outSpatialRef):
    """
    convert points coordinate from old SRS to new SRS
    Args:
        input_x:input points x, list type
        input_y:input points y, list type
        inSpatialRef: object of old SpatialReference
        outSpatialRef:object of new SpatialReference

    Returns:True is successful, False Otherwise

    """
    if len(input_x) != len(input_y):
        basic.outputlogMessage('the count of input x or y is different')
        return False
    ncount = len(input_x)
    if ncount<1:
        basic.outputlogMessage('the count of input x less than 1')
        return False

    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    # print coordTransform
    # start = time.time()
    for i in range(0,ncount):
        # pointX = input_x[i]
        # pointY = input_y[i]
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(input_x[i], input_y[i])
        # transform point
        point.Transform(coordTransform)
        input_x[i] = point.GetX()
        input_y[i] = point.GetY()
    # end = time.time()
    # cost = end - start #time in second
    # print cost

    return True

def convert_points_coordinate_proj4(input_x,input_y,in_proj4, out_proj4):
    inSpatialRef = osr.SpatialReference()
    # inSpatialRef.ImportFromEPSG(inputEPSG)
    inSpatialRef.ImportFromProj4(in_proj4)
    outSpatialRef = osr.SpatialReference()
    # outSpatialRef.ImportFromEPSG(outputEPSG)
    outSpatialRef.ImportFromProj4(out_proj4)
    return convert_points_SpatialRef(input_x,input_y,inSpatialRef,outSpatialRef)

def convert_points_coordinate_epsg(input_x,input_y,in_epsg, out_epsg):
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(in_epsg)
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(out_epsg)
    return convert_points_SpatialRef(input_x,input_y,inSpatialRef,outSpatialRef)

def  convert_points_coordinate(input_x,input_y,inwkt, outwkt ):
    """
    convert points coordinate from old SRS(wkt format) to new SRS(wkt format)
    Args:
        input_x:points x, list type
        input_y:input points y, list type
        inwkt: complete wkt of old SRS
        outwkt: complete wkt of old SRS

    Returns: True is successful, False Otherwise

    """
    # if (isinstance(input_x,list) is False) or (isinstance(input_y,list) is False):
    #     syslog.outputlogMessage('input x or y type error')
    #     return False
    # create coordinate transformation
    inSpatialRef = osr.SpatialReference()
    # inSpatialRef.ImportFromEPSG(inputEPSG)
    inSpatialRef.ImportFromWkt(inwkt)
    outSpatialRef = osr.SpatialReference()
    # outSpatialRef.ImportFromEPSG(outputEPSG)
    outSpatialRef.ImportFromWkt(outwkt)
    return convert_points_SpatialRef(input_x,input_y,inSpatialRef,outSpatialRef)

def get_raster_or_vector_srs_info(spatial_data,format):
    """
    get SRS(Spatial Reference System) information from raster or vector data
    Args:
        spatial_data: the path of raster or vector data
        format: Any of the usual GDAL/OGR forms(complete WKT, PROJ.4, EPSG:n or a file containing the SRS)

    Returns:the string of srs info in special format, False otherwise

    """
    if io_function.is_file_exist(spatial_data) is False:
        return False
    CommandString = 'gdalsrsinfo -o  '+ format +' '+  spatial_data
    result = basic.exec_command_string_output_string(CommandString)
    if result.find('ERROR') >=0:
        return False
    if result.find('Confidence') >= 0:  # new GDAL, with
        print(result)
        tmp = result.split('\n')
        for aa in tmp:
            if aa.find('Confidence') >= 0:
                continue
            if len(aa) > 2:
                result = aa
                break
    result = result.strip()

    return result

def get_raster_or_vector_srs_info_wkt(spatial_data):
    """
    get SRS(Spatial Reference System) information from raster or vector data
    Args:
        spatial_data: the path of raster or vector data

    Returns:the string of srs info in WKT format, False otherwise

    """
    # after gdal > 3.0. there are: wkt_all, wkt1, wkt_simple, wkt_noct, wkt_esri, wkt2, wkt2_2015, wkt2_2018 too complex
    return get_raster_or_vector_srs_info(spatial_data,'wkt')

def get_raster_or_vector_srs_info_proj4(spatial_data):
    """
    get SRS(Spatial Reference System) information from raster or vector data
    Args:
        spatial_data: the path of raster or vector data

    Returns:the string of srs info in proj4 format, False otherwise

    """
    return get_raster_or_vector_srs_info(spatial_data, 'proj4')

def get_raster_or_vector_srs_info_epsg(spatial_data):
    """
    get SRS(Spatial Reference System) information from raster or vector data
    Args:
        spatial_data: the path of raster or vector data

    Returns:the string of srs info in proj4 format, False otherwise

    """
    return get_raster_or_vector_srs_info(spatial_data, 'epsg')


def transforms_vector_srs(shapefile,t_srs,t_file):
    """
    convert vector file to target SRS(Spatial Reference System)
    Args:
        shapefile:input vector file
        t_srs:target SRS(Spatial Reference System)
        t_file:the output target file

    Returns:the output file path is successful, False Otherwise

    """
    if io_function.is_file_exist(shapefile) is False:
        return False
    CommandString = 'ogr2ogr  -t_srs  ' +  t_srs + ' '+ t_file + ' '+ shapefile
    # if result.find('ERROR') >=0 or result.find('FAILURE'):
    #     return False
    return basic.exec_command_string_one_file(CommandString,t_file)

def transforms_raster_srs(rasterfile,t_srs,t_file,x_res,y_res,resample_m='bilinear',
                          o_format='GTiff',compress=None, tiled=None, bigtiff=None):
    """
    convert raster file to target SRS(Spatial Reference System)
    Args:
        rasterfile:input raster file
        t_srs: target SRS(Spatial Reference System)
        t_file:the output target file
        x_res:set output file x-resolution (in target georeferenced units),assigning this value to make sure the resolution would not change in target file
        y_res:set output file y-resolution (in target georeferenced units),assigning this value to make sure the resolution would not change in target file

    Returns:the output file path is successful, False Otherwise

    """
    if io_function.is_file_exist(rasterfile) is False:
        return False
    x_res  = abs(x_res)
    y_res = abs(y_res)
    CommandString = 'gdalwarp  -r %s  -t_srs '%resample_m + t_srs  +' -tr ' +str(x_res)+ ' ' + str(y_res)

    if compress != None:
        CommandString += ' -co ' + 'compress=%s'%compress       # lzw
    if tiled != None:
        CommandString += ' -co ' + 'TILED=%s'%tiled     # yes
    if bigtiff != None:
        CommandString += ' -co ' + 'bigtiff=%s' % bigtiff  # IF_SAFER

    CommandString += ' ' + rasterfile + ' ' + t_file

    return basic.exec_command_string_one_file(CommandString,t_file)

def transforms_raster_srs_to_base_image(rasterfile,baseimage,target_file,x_res,y_res):
    """
    convert raster file to target SRS(Spatial Reference System) of base image
    Args:
        rasterfile:input raster file
        baseimage:a image contains target srs info
        target_file:the output target file
        x_res:set output file x-resolution (in target georeferenced units)
        y_res:set output file y-resolution (in target georeferenced units)

    Returns:the output file path is successful, False Otherwise

    """
    if io_function.is_file_exist(baseimage) is False:
        return False
    target_srs = get_raster_or_vector_srs_info_proj4(baseimage)
    if target_srs is False:
        return False
    return transforms_raster_srs(rasterfile,target_srs,target_file,x_res,y_res)

def meters_to_degrees_onEarth(distance):
    # distance in meters
    return (distance/6371000.0)*180.0/math.pi


if __name__=='__main__':

    # solve this by "conda install gdal -c conda-forge"
    try:
        from osgeo import ogr, osr, gdal
    except:
        raise IOError('ERROR: cannot find GDAL/OGR modules')

    length = len(sys.argv)
    if length == 6:
        rasterfile = sys.argv[1]
        baseimage = sys.argv[2]
        target_file = sys.argv[3]
        x_res = int(sys.argv[4])
        y_res = int(sys.argv[5])
        transforms_raster_srs_to_base_image(rasterfile, baseimage, target_file, x_res, y_res)
    else:
        print ('no Input error, Try to do like this:')
        print ('RSImageProcess.py  ....')
        sys.exit(1)

    pass
