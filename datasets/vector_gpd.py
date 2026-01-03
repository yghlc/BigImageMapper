#!/usr/bin/env python
# Filename: vector_gpd
"""
introduction: similar to vector_features.py, by use geopandas to read and write shapefile

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 08 December, 2019
"""

import os,sys
from optparse import OptionParser


# import these two to make sure load GEOS dll before using shapely
import shapely
from shapely.geometry import mapping # transform to GeJSON format
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon
from shapely.geometry import box
from shapely.geometry import LineString
from shapely.geometry import MultiLineString
from shapely import ops
from shapely.geometry import GeometryCollection
from shapely.ops import unary_union

from shapely.strtree import STRtree
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd

import math
import numpy as np
import time
import random

import basic_src.basic as basic

import basic_src.map_projection as map_projection

from datetime import datetime
from multiprocessing import Pool

from packaging import version
import networkx as nx

import fiona

def check_remove_None_geometries(geometries, gpd_dataframe, file_path=None):
    # Missing and empty geometries, find None geometry, then remove them
    # https://geopandas.org/en/stable/docs/user_guide/missing_empty.html

    # check None in geometries:
    # gpd_dataframe = gpd.read_file(polygon_shp)
    # geometries = shapefile.geometry.values

    idx_list = [ idx for idx, polygon in enumerate(geometries) if polygon is None]
    if len(idx_list) > 0:
        message = 'Warning, %d None geometries, will be removed'%len(idx_list)
        if file_path is not None:
            message += ', file path: %s'%file_path
        for idx in idx_list:
            gpd_dataframe.drop(idx, inplace=True)
            # geometries.drop(idx,inplace=True)     # not working
        basic.outputlogMessage(message)

    # return geometries again after droping some rows
    return gpd_dataframe.geometry.values


def check_remove_None_geometries_file(input_file, output_file):
    """
        Reads a geospatial file, removes rows with None geometries, and saves the cleaned file.

        :param input_file: Path to the input file (GeoJSON, Shapefile, etc.).
        :param output_file: Path to save the output file with cleaned geometries.
        :return: None
        """
    # Read the input file into a GeoDataFrame
    gpd_dataframe = gpd.read_file(input_file)

    # Find rows with None geometries
    none_geometry_indices = gpd_dataframe[gpd_dataframe.geometry.isna()].index

    # Log and remove None geometries
    if len(none_geometry_indices) > 0:
        print(f"Warning: Found {len(none_geometry_indices)} None geometries. Removing them...")

        # Drop rows with None geometries
        gpd_dataframe.drop(none_geometry_indices, inplace=True)
    else:
        print("No None geometries found. No changes made.")

    # Save the cleaned GeoDataFrame to the output file
    gpd_dataframe.to_file(output_file)
    print(f"Cleaned file saved to: {output_file}")

def guess_file_format_extension(file_path):
    _, extension = os.path.splitext(file_path)
    if extension.lower() == '.gpkg':  # GPKG
        return 'GPKG'
    elif extension.lower() == '.shp':  # GPKG
        return 'ESRI Shapefile'
    else:
        raise ValueError('unknown file format from extension: %s'%extension)

def read_polygons_json(polygon_shp, no_json=False):
    '''
    read polyogns and convert to json format
    :param polygon_shp: polygon in projection of EPSG:4326
    :param no_json: True indicate not json format
    :return:
    '''

    # check projection
    shp_args_list = ['gdalsrsinfo', '-o', 'EPSG', polygon_shp]
    epsg_str = basic.exec_command_args_list_one_string(shp_args_list)
    epsg_str = epsg_str.decode().strip()  # byte to str, remove '\n'
    if epsg_str != 'EPSG:4326':
        raise ValueError('Current support shape file in projection of EPSG:4326, but the input has projection of %s'%epsg_str)

    shapefile = gpd.read_file(polygon_shp)
    polygons = shapefile.geometry.values

    # # check invalidity of polygons
    invalid_polygon_idx = []
    # for idx, geom in enumerate(polygons):
    #     if geom.is_valid is False:
    #         invalid_polygon_idx.append(idx + 1)
    # if len(invalid_polygon_idx) > 0:
    #     raise ValueError('error, polygons %s (index start from 1) in %s are invalid, please fix them first '%(str(invalid_polygon_idx),polygon_shp))

    # fix invalid polygons
    polygons = fix_invalid_polygons(polygons)

    if no_json:
        return polygons
    else:
        # convert to json format
        polygons_json = [ mapping(item) for item in polygons]

    return polygons_json

def fix_invalid_polygons(polygons, buffer_size = 0.000001):
    '''
    fix invalid polygon by using buffer operation.
    :param polygons: polygons in shapely format
    :param buffer_size: buffer size
    :return: polygons after checking invalidity
    '''
    invalid_polygon_idx = []
    for idx in range(0,len(polygons)):
        if polygons[idx].is_valid is False:
            invalid_polygon_idx.append(idx + 1)
            polygons[idx] = polygons[idx].buffer(buffer_size)  # trying to solve self-intersection
    if len(invalid_polygon_idx) > 0:
        basic.outputlogMessage('Warning, polygons %s (index start from 1) in are invalid, fix them by the buffer operation '%(str(invalid_polygon_idx)))

    return polygons

def read_lines_gpd(lines_shp):
    shapefile = gpd.read_file(lines_shp)
    lines = shapefile.geometry.values
    # check are lines
    return lines

def read_lines_attributes_list(polygon_shp, field_nameS):
    return read_polygons_attributes_list(polygon_shp, field_nameS, b_fix_invalid_polygon=False)

def find_one_line_intersect_Polygon(polygon, line_list, line_check_list,b_line_only_one_poly):
    for idx, (line, b_checked) in enumerate(zip(line_list,line_check_list)):
        if b_checked and b_line_only_one_poly:
            continue
        if polygon.intersection(line).is_empty is False:
            line_check_list[idx] = True
            return line
    return None

def find_polygon_intersec_polygons(shp_path):

    basic.outputlogMessage('Checking duplicated polygons in %s'%shp_path)

    polygons = read_polygons_gpd(shp_path)

    count = len(polygons)

    for idx, poly in enumerate(polygons):
        for kk in range(idx+1,count):
            inter = poly.intersection(polygons[kk])
            if inter.is_empty is False:
                basic.outputlogMessage('warning, %d th polygon has intersection with %d th polygon'%(idx+1, kk+1))
                # break
    basic.outputlogMessage('finished checking of polygons intersect other polygons')

def read_shape_gpd_to_NewPrj(shp_path, prj_str):
    '''
    read polyogns using geopandas, and reproejct to a projection.
    :param polygon_shp:
    :param prj_str:  project string, like EPSG:4326
    :return:
    '''
    shapefile = gpd.read_file(shp_path)
    # print(shapefile.crs)

    # shapefile  = shapefile.to_crs(prj_str)
    if version.parse(gpd.__version__)  >= version.parse('0.7.0'):
        shapefile = shapefile.to_crs(prj_str)
    else:
        shapefile  = shapefile.to_crs({'init':prj_str})
    # print(shapefile.crs)
    polygons = shapefile.geometry.values
    # fix invalid polygons
    polygons = fix_invalid_polygons(polygons)

    return polygons

def reproject_shapefile(shp_path, prj_str,save_path):
    '''
    reprject a shapefile and save to another path
    :param shp_path: EPSG:4326
    :param prj_str: e.g., EPSG:4326
    :param save_path: save path
    :return:
    '''
    shapefile = gpd.read_file(shp_path)
    # print(shapefile.crs)

    # shapefile  = shapefile.to_crs(prj_str)
    if version.parse(gpd.__version__) >= version.parse('0.7.0'):
        shapefile = shapefile.to_crs(prj_str)
    else:
        shapefile = shapefile.to_crs({'init': prj_str})

    return shapefile.to_file(save_path, driver = 'ESRI Shapefile')

def read_polygons_gpd(polygon_shp, b_fix_invalid_polygon = True):
    '''
    read polyogns using geopandas
    :param polygon_shp: polygon in projection of EPSG:4326
    :param no_json: True indicate not json format
    :return:
    '''

    shapefile = gpd.read_file(polygon_shp)
    polygons = shapefile.geometry.values

    # print('before removing None, %d records'%len(shapefile))
    polygons = check_remove_None_geometries(polygons,shapefile,polygon_shp)
    # print('after removing None, %d records' % len(shapefile))

    # # check invalidity of polygons
    invalid_polygon_idx = []
    # for idx, geom in enumerate(polygons):
    #     if geom.is_valid is False:
    #         invalid_polygon_idx.append(idx + 1)
    # if len(invalid_polygon_idx) > 0:
    #     raise ValueError('error, polygons %s (index start from 1) in %s are invalid, please fix them first '%(str(invalid_polygon_idx),polygon_shp))

    # fix invalid polygons
    if b_fix_invalid_polygon:
        polygons = fix_invalid_polygons(polygons)

    return polygons

def add_attributes_to_shp(shp_path, add_attributes,save_as=None,format='ESRI Shapefile'):
    '''
    add attbibutes to a shapefile
    :param shp_path: the path of shapefile
    :param add_attributes: attributes (dict)
    :return: True if successful, False otherwise
    '''

    shapefile = gpd.read_file(shp_path)
    # print(shapefile.loc[0])   # output the first row

    # get attributes_names
    org_attribute_names = [ key for key in  shapefile.loc[0].keys()]
    # print(org_attribute_names)
    for key in add_attributes.keys():
        if key in org_attribute_names:
            basic.outputlogMessage('warning, field name: %s already in table '
                                       'this will replace the original value'%(key))
        shapefile[key] = add_attributes[key]

    # print(shapefile)
    # save the original file
    if save_as is not None:
        return shapefile.to_file(save_as, driver=format)
    else:
        return shapefile.to_file(shp_path, driver=format)


def read_attribute_values_list(polygon_shp, field_name, out_dtype=None):
    '''
    Read the attribute values from a shapefile column into a list.

    :param polygon_shp: Path to the shapefile.
    :param field_name: The column name to read.
    :param out_dtype: Target data type for the output values (e.g., int, float, str).
    :return: A list containing the attribute values (converted if out_dtype is specified), or None if field is missing.
    '''
    shapefile = gpd.read_file(polygon_shp)

    if field_name in shapefile.keys():
        attribute_values = shapefile[field_name]
        # Convert to target dtype if specified
        if out_dtype is not None:
            converted_values = attribute_values.astype(out_dtype)
            # Optionally, convert NaN to None
            result_list = [v if pd.notnull(v) else None for v in converted_values]
            return result_list
        else:
            return attribute_values.tolist()
    else:
        basic.outputlogMessage('Warning: %s not in the shape file, will return None'%field_name)
        return None

def read_attribute_values_list_2d(polygon_shp, field_nameS):
    '''
    read attribute value (list)
    :param polygon_shp:
    :param field_nameS: a string file name or a list of field_name
    :return: attributes
    '''
    shapefile = gpd.read_file(polygon_shp)

    # read attributes
    if isinstance(field_nameS, str):  # only one field name
        if field_nameS in shapefile.keys():
            attribute_values = shapefile[field_nameS]
            return attribute_values.tolist()
        else:
            basic.outputlogMessage('Warning: %s not in the shape file, get None' % field_nameS)
            return None
    elif isinstance(field_nameS, list):  # a list of field name
        attribute_2d = []
        for field_name in field_nameS:
            if field_name in shapefile.keys():
                attribute_values = shapefile[field_name]
                attribute_2d.append(attribute_values.tolist())
            else:
                basic.outputlogMessage('Warning: %s not in the shape file, get None' % field_nameS)
                attribute_2d.append(None)

        return attribute_2d

def is_field_name_in_shp(polygon_shp, field_name):
    '''
    check a attribute name is in the shapefile
    :param polygon_shp:
    :param field_name:
    :return:
    '''

    # using finoa is much faster than geopanda when the file is large
    with fiona.open(polygon_shp) as src:
        # print("Column names:", list(src.schema['properties'].keys()))
        # To check if 'name' exists:
        return field_name in src.schema['properties']

    # shapefile = gpd.read_file(polygon_shp, rows=0) #  Only reads schema, not data!
    # print(field_name in shapefile.columns)
    # if field_name in shapefile.keys():
    #     return True
    # else:
    #     return False

def read_attribute_name_list(polygon_shp):
    '''
    read all attribute names from a shapefile
    :param polygon_shp:
    :return: a list of attribute names
    '''
    # using finoa is much faster than geopanda when the file is large
    with fiona.open(polygon_shp) as src:
        return list(src.schema['properties'].keys())

def read_polygons_attributes_list(polygon_shp, field_nameS, b_fix_invalid_polygon = True):
    '''
    read polygons and attribute value (list)
    :param polygon_shp:
    :param field_nameS: a string file name or a list of field_name
    :return: Polygons and attributes
    '''
    shapefile = gpd.read_file(polygon_shp)
    polygons = shapefile.geometry.values
    # check None
    polygons = check_remove_None_geometries(polygons,shapefile,polygon_shp)

    # fix invalid polygons
    if b_fix_invalid_polygon:
        polygons = fix_invalid_polygons(polygons)

    # read attributes
    if isinstance(field_nameS,str): # only one field name
        if field_nameS in shapefile.keys():
            attribute_values = shapefile[field_nameS]
            return polygons, attribute_values.tolist()
        else:
            basic.outputlogMessage('Warning: %s not in the shape file, get None' % field_nameS)
            return polygons, None
    elif isinstance(field_nameS,list):  # a list of field name
        attribute_2d = []
        for field_name in field_nameS:
            if field_name in shapefile.keys():
                attribute_values = shapefile[field_name]
                attribute_2d.append(attribute_values.tolist())
            else:
                basic.outputlogMessage('Warning: %s not in the shape file, get None' % field_nameS)
                attribute_2d.append(None)
        return polygons, attribute_2d
    else:
        raise ValueError('unknown type of %s'%str(field_nameS))

def is_two_bound_disjoint(box1, box2):
    # same to the one in raster_io by calling rasterio.coords.disjoint_bounds(box1,box2)
    # but just do not want to import rater_io
    # box: (minx, miny, maxx, maxy)

    # left 1 > right 2 or  right 1 < left 2 or bottom 1 > top 2 or top 1 < bottom 2
    if box1[0] > box2[2] or box1[2] < box2[0] or box1[1] > box2[3] or box1[3] < box2[1]:
        return True
    return False

def get_projection(file_path, format=None):

    # convert the different type, to epsg, proj4, and wkt
    gdf = gpd.read_file(file_path)
    if format is not None:
        if format == 'proj4':
            return gdf.crs.to_proj4()  # string like '+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs',
        elif format == 'wkt':
            return gdf.crs.to_wkt()  # string,  # its OGC WKT representation
        elif format == 'epsg':
            return gdf.crs.to_epsg()  # to epsg code, int, such as 3413
        else:
            raise ValueError('Unknown format: %s' % str(format))

    return gdf.crs


def get_vector_file_bounding_box(file_path):
    # return bounding box of all geometryies ((minx, miny, maxx, maxy))
    shapefile = gpd.read_file(file_path)
    return shapefile.total_bounds

def get_polygon_bounding_box(polygon):
    # return the bounding box of a shapely polygon (minx, miny, maxx, maxy)
    return polygon.bounds


def get_polygon_centroid_lat_lon(in_shp):
    geometries = read_shape_gpd_to_NewPrj(in_shp,'EPSG:4326')
    # Get centroids and extract (lat, lon)
    centroids = geometries.centroid
    lat_lon_list = [(pt.y, pt.x) for pt in centroids]
    return lat_lon_list


def get_polygon_centroid(polygon):
    # return the geometric center of a polygon
    return polygon.centroid

def get_polygon_representative_point(polygon):
    # centroid not always inside a polygon,
    # use representative_point to get a point alway inside the polygon, not in general the same as the centroid
    return polygon.representative_point()

def get_polygon_envelope_xy(polygon):
    # get polygon envelope x,y coordinates
    # polygon, shapely polygon
    # output: x: a list of x0 to x4  y: a list of y0 to y4      # the last one is the same as the first one.
    polygon_env = polygon.envelope
    x, y = polygon_env.exterior.coords.xy
    return x,y

def get_box_polygon_leftUp_rightDown(box_polygon):
    # get box left-up and right-botton x,y coordinates
    # box_polygon, shapely polygon
    # output: (x1, y1, x2, y2)      # the last one is the same as the first one.
    x, y = box_polygon.exterior.coords.xy
    return (x[0], y[0], x[2], y[2])

def remove_polygon_equal(shapefile,field_name, expect_value, b_equal, output):
    '''
    remove polygons the the attribute value is not equal to a specific value
    :param shapefile:
    :param field_name:
    :param threshold:
    :param b_equal: if True, remove records not equal to expect_value, otherwise, remove the one equal to expect_value
    :param output:
    :return:
    '''

    shapefile = gpd.read_file(shapefile)

    remove_count = 0

    for idx,row in shapefile.iterrows():

        # polygon = row['geometry']
        # go through post-processing to decide to keep or remove it
        if b_equal:
            if row[field_name] != expect_value:
                shapefile.drop(idx, inplace=True)
                remove_count += 1
        else:
            if row[field_name] == expect_value:
                shapefile.drop(idx, inplace=True)
                remove_count += 1

    basic.outputlogMessage('remove %d polygons based on %s, remain %d ones saving to %s' %
                           (remove_count, field_name, len(shapefile.geometry.values), output))
    # save results
    shapefile.to_file(output, driver='ESRI Shapefile')

def remove_polygon_time_index(shapefile,field_name, time_count, output):
    '''
    remove polygons if the time index is not monotonically increasing and not follow the pattern
    :param shapefile:
    :param field_name:
    :param time_count:
    :param output:
    :return:
    '''
    remove_count = 0
    shapefile = gpd.read_file(shapefile)

    for idx, row in shapefile.iterrows():
        idx_string = row[field_name]
        num_list = [int(item) for item in idx_string.split('_')]
        # the number list should be one of the pattern: 0, 1, 2...n  or 1, 2,...n or n, not only monotonically increasing
        pattern_int = [str(item) for item in range(num_list[0],time_count)]
        pattern_str = '_'.join(pattern_int)
        # if np.all(np.diff(num_list) >= 1):
        if idx_string == pattern_str:
            pass
        else:
            shapefile.drop(idx, inplace=True)
            remove_count += 1

    basic.outputlogMessage('remove %d polygons based on %s, remain %d ones saving to %s' %
                           (remove_count, field_name, len(shapefile.geometry.values), output))
    # save results
    return shapefile.to_file(output, driver='ESRI Shapefile')

def remove_polygon_index_string(shapefile,field_name, index_list, output):
    '''
    remove polygons the the attribute value is not equal to a specific value
    :param shapefile:
    :param field_name:
    :param threshold:
    :param b_equal: if True, remove records not equal to expect_value, otherwise, remove the one equal to expect_value
    :param output:
    :return:
    '''
    if len(index_list) < 1:
        raise ValueError('Wrong input index_list, it size is zero')

    shapefile = gpd.read_file(shapefile)

    remove_count = 0

    for idx,row in shapefile.iterrows():

        # polygon = row['geometry']
        # go through post-processing to decide to keep or remove it
        idx_string = row[field_name]
        num_list =  [ int(item) for item in idx_string.split('_')]

        # if all the index in index_list found in num_list, then keep it, otherwise, remove it
        for index in index_list:
            if index not in num_list:
                shapefile.drop(idx, inplace=True)
                remove_count += 1
                break

    basic.outputlogMessage('remove %d polygons based on %s, remain %d ones saving to %s' %
                           (remove_count, field_name, len(shapefile.geometry.values), output))
    # save results
    shapefile.to_file(output, driver='ESRI Shapefile')

def remove_polygons_not_in_range(shapefile,field_name, min_value, max_value,output):
    '''
    remove polygon not in range (min, max]
    :param shapefile:
    :param field_name:
    :param min_value:
    :param max_value:
    :param output:
    :return:
    '''
    # read polygons as shapely objects
    shapefile = gpd.read_file(shapefile)

    remove_count = 0

    for idx, row in shapefile.iterrows():

        # polygon = row['geometry']
        # go through post-processing to decide to keep or remove it
        if row[field_name] < min_value or row[field_name] >= max_value:
            shapefile.drop(idx, inplace=True)
            remove_count += 1

    if len(shapefile.geometry.values) < 1:
        basic.outputlogMessage('remove %d polygons based on %s, remain %d ones, no saved files' %
                               (remove_count, field_name, len(shapefile.geometry.values)))
        return False
    else:
        basic.outputlogMessage('remove %d polygons based on %s, remain %d ones saving to %s' %
                           (remove_count, field_name, len(shapefile.geometry.values), output))
    # save results
    shapefile.to_file(output, driver='ESRI Shapefile')

def remove_polygons_based_values(shapefile,value_list, threshold, bsmaller,output):
    '''
    remove polygons based on attribute values
    :param shapefile:
    :param value_list: values for removing polygons, its number should be the same polygon numbers in shapefile
    :param threshold:
    :param bsmaller:
    :param output:
    :return:
    '''
    # read polygons as shapely objects
    shapefile = gpd.read_file(shapefile)
    org_count = len(shapefile)

    # for (idx,row), value in zip(shapefile.iterrows(),value_list):
    #     if bsmaller:
    #         if value < threshold:
    #             shapefile.drop(idx, inplace=True)
    #             remove_count += 1
    #     else:
    #         if value >= threshold:
    #             shapefile.drop(idx, inplace=True)
    #             remove_count += 1

    value_array = np.array(value_list)
    if bsmaller:
        # remove small values, so keep the bigger ones
        shapefile = shapefile[value_array >= threshold]
    else:
        # remove small values, so keep the small ones
        shapefile = shapefile[value_array < threshold]

    remove_count = org_count - len(shapefile)

    if len(shapefile.geometry.values) < 1:
        basic.outputlogMessage('remove %d polygons based on a list of values, remain %d ones, no saved files' %
                               (remove_count, len(shapefile.geometry.values)))
        return False
    else:
        basic.outputlogMessage('remove %d polygons, remain %d ones saving to %s' %
                           (remove_count, len(shapefile.geometry.values), output))
    # save results
    shapefile.to_file(output, driver='ESRI Shapefile')

def remove_polygons(shapefile,field_name, threshold, bsmaller,output):
    '''
    remove polygons based on attribute values
    :param shapefile:
    :param field_name:
    :param threshold:
    :param bsmaller:
    :param output:
    :return:
    '''
    # another version
    # operation_obj = shape_opeation()
    # if operation_obj.remove_shape_baseon_field_value(shapefile, output, field_name, threshold, smaller=bsmaller) is False:
    #     return False

    # read polygons as shapely objects
    shapefile = gpd.read_file(shapefile)

    remove_count = 0

    for idx,row in shapefile.iterrows():

        # polygon = row['geometry']
        # go through post-processing to decide to keep or remove it

        if bsmaller:
            if row[field_name] < threshold:
                shapefile.drop(idx, inplace=True)
                remove_count += 1
        else:
            if row[field_name] >= threshold:
                shapefile.drop(idx, inplace=True)
                remove_count += 1

    if len(shapefile.geometry.values) < 1:
        basic.outputlogMessage('remove %d polygons based on %s, remain %d ones, no saved files' %
                               (remove_count, field_name, len(shapefile.geometry.values)))
        return False
    else:
        basic.outputlogMessage('remove %d polygons based on %s, remain %d ones saving to %s' %
                           (remove_count, field_name, len(shapefile.geometry.values), output))
    # save results
    shapefile.to_file(output, driver='ESRI Shapefile')

def calculate_polygon_shape_info(polygon_shapely):
    '''
    calculate the shape information of a polygon, including area, perimeter, circularity,
    WIDTH and HEIGHT based on minimum_rotated_rectangle,
    :param polygon_shapely: a polygon (shapely object)
    :return:
    '''
    shape_info = {}
    shape_info['INarea'] = polygon_shapely.area
    shape_info['INperimete']  = polygon_shapely.length

    if polygon_shapely.is_empty:
        shape_info['circularit'] = 0
    else:
        # circularity
        circularity = (4 * math.pi *  polygon_shapely.area / polygon_shapely.length** 2)
        shape_info['circularit'] = circularity

    if polygon_shapely.is_empty:
        shape_info['WIDTH'] = 0
        shape_info['HEIGHT'] = 0
        shape_info['ratio_w_h'] = 0
    else:
        minimum_rotated_rectangle = polygon_shapely.minimum_rotated_rectangle

        points = list(minimum_rotated_rectangle.boundary.coords)
        point1 = Point(points[0])
        point2 = Point(points[1])
        point3 = Point(points[2])
        width = point1.distance(point2)
        height = point2.distance(point3)

        shape_info['WIDTH'] = width
        shape_info['HEIGHT'] = height
        if width > height:
            shape_info['ratio_w_h'] = height / width
        else:
            shape_info['ratio_w_h'] = width / height

    #added number of holes
    if polygon_shapely.geom_type == 'Polygon':
        shape_info['hole_count'] = len(list(polygon_shapely.interiors))
    else:
        polygons = MultiPolygon_to_polygons(0, polygon_shapely)
        hole_count = 0
        for poly in polygons:
            hole_count += len(list(poly.interiors))
        shape_info['hole_count'] = hole_count

    return shape_info

# convert the list from calculate_polygon_shape_info to a dict for saving to shapefile.
def list_to_dict(list_dict):
    out_dict = {}
    for dict_obj in list_dict:
        for key in dict_obj.keys():
            if key in out_dict.keys():
                out_dict[key].append(dict_obj[key])
            else:
                out_dict[key] = [dict_obj[key]]
    return out_dict

def save_shapefile_subset_as_valueInlist(org_shp, save_path, field_name, value_list, format='ESRI Shapefile'):
    '''
    save a subset of vector file based on values in a column (field)
    :param org_shp: original shapefile
    :param save_path: save path
    :param field_name: the file name
    :param value_list: a list of values, if the value of the column is in this list, then will save the record
    :param format:
    :return:
    '''
    #
    shapefile = gpd.read_file(org_shp)
    filtered_shapefile = shapefile[shapefile[field_name].isin(value_list)]
    # change format
    guess_format = guess_file_format_extension(save_path)
    if guess_format != format:
        basic.outputlogMessage('warning, the format (%s) associated with file extension is different with the input one (%s)'%
                               (guess_format,format))
        format = guess_format
    filtered_shapefile.to_file(save_path, driver=format)
    basic.outputlogMessage('save subset (%d geometry) of shapefile to %s'%(len(filtered_shapefile),save_path))

def save_shapefile_subset_as(data_poly_indices, org_shp, save_path,format='ESRI Shapefile'):
    '''
    save subset of shapefile
    :param data_poly_indices: polygon index
    :param org_shp: orignal shapefile
    :param save_path: save path
    :return: True if successful
    '''
    if len(data_poly_indices) < 1:
        raise ValueError('no input index')

    save_count = len(data_poly_indices)
    shapefile = gpd.read_file(org_shp)
    # nrow, ncol = shapefile.shape

    # selected_list = [False]*nrow
    # for idx in data_poly_indices:
    #     selected_list[idx] = True

    # shapefile_sub = shapefile[selected_list]
    shapefile_sub = shapefile.iloc[data_poly_indices]
    # change format
    guess_format = guess_file_format_extension(save_path)
    if guess_format != format:
        basic.outputlogMessage('warning, the format (%s) associated with file extension is different with the input one (%s)'%
                               (guess_format,format))
        format = guess_format

    shapefile_sub.to_file(save_path, driver=format)
    basic.outputlogMessage('save subset (%d geometry) of shapefile to %s'%(save_count,save_path))

    return True

def save_polygons_to_files(data_frame, geometry_name, wkt_string, save_path,format='ESRI Shapefile'):
    '''
    :param data_frame: include polygon list and the corresponding attributes
    :param geometry_name: dict key for the polgyon in the DataFrame
    :param wkt_string: wkt string (projection)
    :param save_path: save path
    :param format: use ESRI Shapefile or "GPKG" (GeoPackage)
    :return:
    '''
    # data_frame[geometry_name] = data_frame[geometry_name].apply(wkt.loads)
    poly_df = gpd.GeoDataFrame(data_frame, geometry=geometry_name)
    poly_df.crs = wkt_string # or poly_df.crs = {'init' :'epsg:4326'}
    if format=='ESRI Shapefile' and save_path.endswith('.shp') is False:
        basic.outputlogMessage('Warning, the extension of the save file is not shp, adding .shp')
        save_path = os.path.splitext(save_path)[0] + ".shp"

    poly_df.to_file(save_path, driver=format)

    return True

def save_lines_to_files(data_frame, geometry_name, wkt_string, save_path,format='ESRI Shapefile'):
    return save_polygons_to_files(data_frame, geometry_name, wkt_string, save_path,format=format)

def save_points_to_file(data_frame, geometry_name, wkt_string, save_path,format='ESRI Shapefile'):
    return save_polygons_to_files(data_frame, geometry_name, wkt_string, save_path,format=format)

def remove_narrow_parts_of_a_polygon(shapely_polygon, rm_narrow_thr):
    '''
    try to remove the narrow (or thin) parts of a polygon by using buffer opeartion
    :param shapely_polygon: a shapely object, Polygon or MultiPolygon
    :param rm_narrow_thr: how narrow
    :return: the shapely polygon, multipolygons or polygons
    '''

    # A positive distance has an effect of dilation; a negative distance, erosion.
    # object.buffer(distance, resolution=16, cap_style=1, join_style=1, mitre_limit=5.0)

    enlarge_factor = 1.6
    # can return multiple polygons
    # remain_polygon_parts = shapely_polygon.buffer(-rm_narrow_thr)
    # remain_polygon_parts = shapely_polygon.buffer(-rm_narrow_thr).buffer(rm_narrow_thr * enlarge_factor)
    remain_polygon_parts = shapely_polygon.buffer(-rm_narrow_thr).buffer(rm_narrow_thr * enlarge_factor).intersection(shapely_polygon)

    return remain_polygon_parts

def remove_narrow_parts_of_polygons_shp_NOmultiPolygon(input_shp,out_shp,rm_narrow_thr):
    # read polygons as shapely objects
    shapefile = gpd.read_file(input_shp)

    attribute_names = None
    new_polygon_list = []
    polygon_attributes_list = []  # 2d list

    for idx, row in shapefile.iterrows():
        if idx==0:
            attribute_names = row.keys().to_list()[:-1]  # the last one is 'geometry'
        print('removing narrow parts of %dth polygon (total: %d)'%(idx+1,len(shapefile.geometry.values)))
        shapely_polygon = row['geometry']
        if shapely_polygon.is_valid is False:
            shapely_polygon = shapely_polygon.buffer(0.000001)
            basic.outputlogMessage('warning, %d th polygon is is_valid, fix it by the buffer operation'%idx)
        out_geometry = remove_narrow_parts_of_a_polygon(shapely_polygon, rm_narrow_thr)
        # if out_polygon.is_empty is True:
        #     print(idx, out_polygon)
        if out_geometry.is_empty is True:
            basic.outputlogMessage('Warning, remove %dth (0 index) polygon in %s because it is empty after removing narrow parts'%
                                   (idx, os.path.basename(input_shp)))
            # continue, don't save
            # shapefile.drop(idx, inplace=True),
        else:
            out_polygon_list = MultiPolygon_to_polygons(idx, out_geometry)
            if len(out_polygon_list) < 1:
                continue
            new_polygon_list.extend(out_polygon_list)
            attributes = [row[key] for key in attribute_names]
            for idx in range(len(out_polygon_list)):
                # copy the attributes (Not area and perimeter, etc)
                polygon_attributes_list.append(attributes)        # last one is 'geometry'
            # copy attributes

    if len(new_polygon_list) < 1:
        basic.outputlogMessage('Warning, no polygons in %s'%input_shp)
        return False

    save_polyons_attributes = {}
    for idx, attribute in enumerate(attribute_names):
        # print(idx, attribute)
        values = [item[idx] for item in polygon_attributes_list]
        save_polyons_attributes[attribute] = values

    save_polyons_attributes["Polygons"] = new_polygon_list
    polygon_df = pd.DataFrame(save_polyons_attributes)

    basic.outputlogMessage('After removing the narrow parts, obtaining %d polygons'%len(new_polygon_list))
    print(out_shp, isinstance(out_shp,list))
    basic.outputlogMessage('will be saved to %s'%out_shp)
    wkt_string = map_projection.get_raster_or_vector_srs_info_wkt(input_shp)
    return save_polygons_to_files(polygon_df, 'Polygons', wkt_string, out_shp)

def remove_narrow_parts_of_polygons_shp(input_shp,out_shp,rm_narrow_thr):
    # read polygons as shapely objects
    shapefile = gpd.read_file(input_shp)

    attribute_names = None
    new_polygon_list = []
    polygon_attributes_list = []  # 2d list

    for idx, row in shapefile.iterrows():
        if idx==0:
            attribute_names = row.keys().to_list()[:-1]  # the last one is 'geometry'
        print('removing narrow parts of %dth polygon (total: %d)'%(idx+1,len(shapefile.geometry.values)))
        shapely_polygon = row['geometry']
        out_polygon = remove_narrow_parts_of_a_polygon(shapely_polygon, rm_narrow_thr)
        # if out_polygon.is_empty is True:
        #     print(idx, out_polygon)
        if out_polygon.is_empty is True:
            basic.outputlogMessage('Warning, remove %dth (0 index) polygon in %s because it is empty after removing narrow parts'%
                                   (idx, os.path.basename(input_shp)))
            # continue, don't save
            # shapefile.drop(idx, inplace=True),
        else:
            new_polygon_list.append(out_polygon)
            attributes = [row[key] for key in attribute_names]
            polygon_attributes_list.append(attributes)        # last one is 'geometry'
            # copy attributes

    save_polyons_attributes = {}
    for idx, attribute in enumerate(attribute_names):
        # print(idx, attribute)
        values = [item[idx] for item in polygon_attributes_list]
        save_polyons_attributes[attribute] = values

    save_polyons_attributes["Polygons"] = new_polygon_list
    polygon_df = pd.DataFrame(save_polyons_attributes)

    basic.outputlogMessage('After removing the narrow parts, obtaining %d polygons'%len(new_polygon_list))
    print(out_shp, isinstance(out_shp,list))
    basic.outputlogMessage('will be saved to %s'%out_shp)
    wkt_string = map_projection.get_raster_or_vector_srs_info_wkt(input_shp)
    return save_polygons_to_files(polygon_df, 'Polygons', wkt_string, out_shp)

def polygons_to_a_MultiPolygon(polygon_list):
    if isinstance(polygon_list,list) is False:
        raise ValueError('the input is a not list')
    if len(polygon_list) < 1:
        raise ValueError('There is no polygon in the input')
    return MultiPolygon(polygon_list)

def MultiPolygon_to_polygons(idx, multiPolygon, attributes=None):
    ''''''
    if version.parse(shapely.__version__) >= version.parse("2.0.0"):
        geometry_values = multiPolygon.geoms
    else:
        geometry_values = multiPolygon

    if multiPolygon.geom_type == 'GeometryCollection':
        polygons = []
        # print(multiPolygon)
        # geometries = list(multiPolygon)
        # print(geometries)
        for geometry in geometry_values:
            # print(geometry)
            if geometry.geom_type == 'Polygon':
                polygons.append(geometry)
            elif geometry.geom_type == 'MultiPolygon':
                polygons.extend(list(geometry))
            else:
                basic.outputlogMessage("Warning, abandon a %s derived from the %d th polygon "%(geometry.geom_type,idx))

    elif multiPolygon.geom_type == 'MultiPolygon':
        polygons = list(geometry_values)
    elif multiPolygon.geom_type == 'Polygon':
        polygons = [multiPolygon]
    else:
        raise ValueError('Currently, only support Polygon and MultiPolygon, but input is %s' % multiPolygon.geom_type)


    # # TODO: calculate new information each polygon
    # polygon_attributes_list = []        # 2D list for polygons
    # for p_idx, polygon in enumerate(polygons):
    #     # print(polyon.area)
    #     # calculate area, circularity, oriented minimum bounding box
    #     polygon_shape = calculate_polygon_shape_info(polygon)
    #     if idx == 0 and p_idx == 0:
    #         [attribute_names.append(item) for item in polygon_shape.keys()]
    #
    #     [polygon_attributes.append(polygon_shape[item]) for item in polygon_shape.keys()]
    #     polygon_attributes_list.append(polygon_attributes)
    #     polygon_list.append(polygon)

    return polygons

def fill_holes_in_polygons_shp(in_shp, out_shp):
    '''
    fill all holes in polygons in a shape file
    :param in_shp:
    :param out_shp:
    :return:
    '''

    # read polygons as shapely objects
    shapefile = gpd.read_file(in_shp)

    for idx, row in shapefile.iterrows():

        poly = row['geometry']
        if poly.type == 'MultiPolygon':
            out_polygons = MultiPolygon_to_polygons(idx, poly)
            basic.outputlogMessage('Warning, %d geometry is "MultiPolygon", convert it to Polygons and copy attributes' % idx)
            for ii, new_poly in enumerate(out_polygons):
                new_poly = fill_holes_in_a_polygon(new_poly)
                row['geometry'] = new_poly
                # replace the first one, appends others
                if ii == 0:
                    shapefile.iloc[idx] = row
                else:
                    shapefile = shapefile.append(row)

        else:
            new_poly = fill_holes_in_a_polygon(poly)
            row['geometry'] = new_poly
            # replace the row
            shapefile.iloc[idx] = row

    # save results
    shapefile.to_file(out_shp, driver='ESRI Shapefile')

    return True

def fill_holes_in_a_polygon(polygon):
    '''
    fill holes in a polygon
    :param polygon: a polygon object (shapely)
    :return:
    '''
    # about MultiPolygon https://stackoverflow.com/questions/48082553/convert-multipolygon-to-polygon-in-python
    if polygon.interiors:
        return Polygon(list(polygon.exterior.coords))
    else:
        return polygon

def get_poly_index_within_extent(polygon_list, extent_poly, min_overlap_area=None):
    '''
    get id of polygons intersecting with an extent
    (may also consider using ogr2ogr to crop the shapefile, also can use remove functions)
    :param polygon_list: polygons list (polygon is in shapely format)
    :param extent_poly: extent polygon (shapely format)
    :param min_overlap_area: if the overlap area is too smaller, than ignore it
    :return: id list
    '''
    idx_list = []
    for idx, poly in enumerate(polygon_list):
        inter = extent_poly.intersection(poly)
        if inter.is_empty is False:
            if min_overlap_area is not None:
                if inter.area < min_overlap_area:
                    continue
            idx_list.append(idx)

    return idx_list

def get_poly_within_extent(polygon_list, extent_poly, min_overlap_area=None, polygon_boxes=None):
    '''
    get polygons intersecting with an extent
    (may also consider using ogr2ogr to crop the shapefile, also can use remove functions)
    :param polygon_list: polygons list (polygon is in shapely format)
    :param extent_poly: extent polygon (shapely format)
    :param min_overlap_area: if the overlap area is too smaller, than ignore it
    :param polygon_boxes: a list of polygon bound (minx, miny, maxx, maxy), to avoid unnecessary intersection calculation
    :return: id list
    '''
    out_polygon_list = []
    if polygon_boxes is not None:
        # update polygons list
        ext_box = get_polygon_bounding_box(extent_poly)
        polygon_list = [poly for poly, box in zip(polygon_list,polygon_boxes) if is_two_bound_disjoint(box,ext_box) is False]

    for idx, poly in enumerate(polygon_list):
        inter = extent_poly.intersection(poly)
        if inter.is_empty is False:
            if min_overlap_area is not None:
                if inter.area < min_overlap_area:
                    continue
            out_polygon_list.append(poly)

    return out_polygon_list

def convert_image_bound_to_shapely_polygon(img_bound_box):
    # convert bounding box  to shapely polygon
    # img_bound_box: bounding box: (left, bottom, right, top) read from rasterio
    letftop1 = (img_bound_box[0],img_bound_box[3])
    righttop1 = (img_bound_box[2],img_bound_box[3])
    rightbottom1 = (img_bound_box[2],img_bound_box[1])
    leftbottom1 = (img_bound_box[0],img_bound_box[1])
    polygon = Polygon([letftop1, righttop1,rightbottom1,leftbottom1])
    return polygon

def convert_bounds_to_polygon(bounds):
    # bounding box: (left, bottom, right, top)
    letftop1 = (bounds[0],bounds[3])
    righttop1 = (bounds[2],bounds[3])
    rightbottom1 = (bounds[2],bounds[1])
    leftbottom1 = (bounds[0],bounds[1])
    polygon = Polygon([letftop1, righttop1,rightbottom1,leftbottom1])
    return polygon

def get_overlap_area_two_boxes(box1, box2, buffer=None):
    '''
    get overlap areas of two box
    :param box1: bounding box: (left, bottom, right, top)
    :param box2: bounding box: (left, bottom, right, top)
    :return: area
    '''
    # print(box1,box1[0],box1[1],box1[2],box1[3])
    letftop1 = (box1[0],box1[3])
    righttop1 = (box1[2],box1[3])
    rightbottom1 = (box1[2],box1[1])
    leftbottom1 = (box1[0],box1[1])
    polygon1 = Polygon([letftop1, righttop1,rightbottom1,leftbottom1])
    # print(polygon1)

    letftop2 = (box2[0],box2[3])
    righttop2 = (box2[2],box2[3])
    rightbottom2 = (box2[2],box2[1])
    leftbottom2 = (box2[0],box2[1])
    polygon2 = Polygon([letftop2, righttop2,rightbottom2,leftbottom2])

    if buffer is not None:
        polygon1 = polygon1.buffer(buffer)
        polygon2 = polygon2.buffer(buffer)

    inter = polygon1.intersection(polygon2)
    # inter = polygon2.intersection(polygon1)
    if inter.is_empty:
        return 0
    if inter.geom_type == 'Polygon':
        return inter.area
    else:
        raise ValueError('need more support of the type: %s'% str(inter.geom_type))

def is_two_polygons_connected(polygon1, polygon2):
    intersection = polygon1.intersection(polygon2)
    if intersection.is_empty:
        return False
    return True

def find_adjacent_polygons(in_polygon, polygon_list, buffer_size=None, Rtree=None):
    # find adjacent polygons
    # in_polygon is the center polygon
    # polygon_list is a polygon list without in_polygon

    if buffer_size is not None:
        center_poly =  in_polygon.buffer(buffer_size)
    else:
        center_poly = in_polygon

    if len(polygon_list) < 1:
        return []

    # https://shapely.readthedocs.io/en/stable/manual.html#str-packed-r-tree
    if Rtree is None:
        tree = STRtree(polygon_list)
    else:
        tree = Rtree
    # query: Returns a list of all geometries in the strtree whose extents intersect the extent of geom.
    # This means that a subsequent search through the returned subset using the desired binary predicate
    # (eg. intersects, crosses, contains, overlaps) may be necessary to further filter the results according
    # to their specific spatial relationships.

    # https://www.geeksforgeeks.org/introduction-to-r-tree/
    # R-trees are faster than Quad-trees for Nearest Neighbour queries while for window queries, Quad-trees are faster than R-trees


    # quicker than check one by one
    # adjacent_polygons = [item for item in tree.query(center_poly) if item.intersection(center_poly) ]
    # t0= time.time()
    adjacent_polygons = [item for item in tree.query(center_poly) if item.intersects(center_poly) or item.touches(center_poly) ]
    # adjacent_poly_idx = [polygon_list.index(item) for item in adjacent_polygons ]
    # print('cost %f seconds'%(time.time() - t0))

    # adjacent_polygons = []
    # adjacent_poly_idx = []
    # for idx, poly in enumerate(polygon_list):
    #     if is_two_polygons_connected(poly, center_poly):
    #         adjacent_polygons.append(poly)
    #         adjacent_poly_idx.append(idx)

    # print(datetime.now(), 'find %d adjacent polygons' % len(adjacent_polygons))

    return adjacent_polygons

def find_adjacent_polygons_from_sub(c_polygon_idx, polygon_list,polygon_boxes,  start_idx, end_idx):

    check_polygons = [polygon_list[j] for j in range(start_idx, end_idx)
                      if is_two_bound_disjoint(polygon_boxes[c_polygon_idx],polygon_boxes[j]) is False ]
    adj_polygons = find_adjacent_polygons(polygon_list[c_polygon_idx], check_polygons)
    # change polygon index to the entire polygons list
    adj_poly_idxs = [polygon_list.index(item) for item in adj_polygons]
    return c_polygon_idx, adj_polygons, adj_poly_idxs


def split_large_group_iterative(group, polygon_list, overlap_threshold, group_max_area, b_verbose=False):
    """
    Splits a group of polygons into smaller subgroups iteratively if the merged area exceeds the max area.

    Parameters:
        group (list): Indices of polygons in the group.
        polygon_list (list): List of all polygons.
        group_max_area (float): Maximum area allowed for each subgroup.

    Returns:
        list of lists: Smaller groups where each group's merged area is below the max area.
    """

    final_subgroups = []

    merged_polygon = unary_union([polygon_list[i] for i in group])
    if merged_polygon.area > group_max_area:
        # build a graph
        group_polygons = [ polygon_list[i] for i in group ]
        tree = STRtree(group_polygons)

        # Create a small graph for overlaps within the input group
        G = nx.Graph()
        G.add_nodes_from(range(len(group_polygons)))

        for i, poly in enumerate(group_polygons):
            potential_overlaps = tree.query(poly)

            for j in potential_overlaps:
                if i >= j:
                    continue
                # Calculate intersection area
                intersection_area = poly.intersection(group_polygons[j]).area
                if intersection_area > overlap_threshold:
                    G.add_edge(i, j, weight=intersection_area)

        # Sort edges by weight, descending, sorted_edges is a list
        sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
        # print(sorted_edges)
        # print(type(sorted_edges))
        # for item in sorted_edges:
        #     print(item)

        # Iteratively find subgroups that meet the area constraint
        sub_group_list = []
        while G.number_of_nodes() > 0:
            # Start with the edge with the highest weight
            if not sorted_edges:
                break  # Exit if no edges are left
            n1, n2, data = sorted_edges.pop(0)
            # print(n1, n2, data)
            G.remove_edge(n1, n2)

            # Create an initial group with the two nodes
            a_group = {n1, n2}
            merged_poly = unary_union([group_polygons[i] for i in a_group])
            merged_poly_area = merged_poly.area

            while merged_poly_area <= group_max_area:
                if not sorted_edges:
                    break
                found_valid_edge = False
                for idx, (u, v, data) in enumerate(sorted_edges):
                    if u in a_group or v in a_group:
                        found_valid_edge = True
                        a_group.update([u,v])
                        _ = sorted_edges.pop(idx)
                        G.remove_edge(u, v)
                        break
                if found_valid_edge is False:
                    break
                merged_poly = unary_union([group_polygons[i] for i in a_group])
                merged_poly_area = merged_poly.area

            sub_group_list.append(a_group)

            # remove nodes
            # isolated_nodes = [node for node in G.nodes if G.degree(node) == 0]
            G.remove_nodes_from(list(a_group))
            # update sorted_edges
            sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)

        # convert polygons index from group_polygons to polygon_list
        for sub_group in sub_group_list:
            final_subgroups.append([group[item] for item in sub_group])

        if b_verbose:
            print(group)
            print(f'divide a group (size {len(group)}) into {len(final_subgroups)} sub-groups')
            print(final_subgroups)

    else:
        final_subgroups.append(group)


    return final_subgroups

def group_overlap_polygons(polygon_list, overlap_threshold=10000, group_max_area=50000,b_verbose=False):
    """
       Groups polygons that overlap by at least a specified intersection area and limits the size of each group.

       Parameters:
           polygon_list (list of shapely.geometry.Polygon): List of polygons to group.
           overlap_threshold (float): Minimum intersection area to consider two polygons as overlapping.
           group_max_area (float): Maximum area allowed for merged groups. Groups exceeding this will be split.

       Returns:
           list of lists: Groups of polygons where each group contains indices of overlapping polygons,
                          and no group exceeds the specified maximum area.
       """

    # Handle empty input
    if not polygon_list:
        return []

    # Ensure all polygons are valid
    if not all(poly.is_valid for poly in polygon_list):
        raise ValueError("All polygons in the list must be valid Shapely geometries.")

    # Build spatial index
    tree = STRtree(polygon_list)

    # Create graph
    G = nx.Graph()
    G.add_nodes_from(range(len(polygon_list)))

    # Check overlaps
    for i, poly in enumerate(polygon_list):
        # Query nearby polygons
        potential_overlaps = tree.query(poly)

        for j in potential_overlaps:
            if i >= j:  # Avoid duplicate comparisons
                continue

            # Calculate intersection area
            intersection_area = poly.intersection(polygon_list[j]).area

            # Add edge if overlap exceeds threshold
            if intersection_area >= overlap_threshold:
                G.add_edge(i, j)

    # Find connected components (initial groups)
    initial_groups = list(nx.connected_components(G))
    print(datetime.now(), 'Done: getting initial groups')

    # Process each group to ensure it meets the area constraint
    final_groups = []
    for group in initial_groups:
        # print(group)
        group = sorted(list(map(int, group)))  # Convert set to list for easier handling
        # print(group)
        if len(group) < 2:
            final_groups.append(group)
        else:
            # to check and split if the group is too large
            final_groups.extend(split_large_group_iterative(group, polygon_list, overlap_threshold, group_max_area,b_verbose=b_verbose))

        # testing
        # break

    return final_groups

def build_adjacent_map_of_polygons(polygons_list, process_num = 1):
    """
    build an adjacent matrix of the tou
    :param polygons_list: a list contains all the shapely (not pyshp) polygons
    :return: a matrix storing the adjacent (shared points) for all polygons
    """

    # another implement is in the vector_features.py,
    # here, we implement the calculation parallel to improve the efficiency.

    # the input polgyons are all valid.

    polygons_list = [item for item in polygons_list]  # GeometryArray to list
    polygon_count = len(polygons_list)
    if polygon_count < 2:
        basic.outputlogMessage('error, the count of polygon is less than 2')
        return False

    # # https://shapely.readthedocs.io/en/stable/manual.html#str-packed-r-tree
    # tree = STRtree(polygons_list)
    polygon_boxes = [ get_polygon_bounding_box(item) for item in polygons_list]

    # this would take a lot of memory if they are many polyton, such as more than 10 000
    ad_matrix = np.zeros((polygon_count, polygon_count),dtype=np.int8)

    if process_num == 1:
        for i in range(0,polygon_count):
            t0 = time.time()
            # if i%100 == 0:
            #     start_idx = i+1
            #     check_polygons = [polygons_list[j] for j in range(start_idx, polygon_count)]
            #     tree = STRtree(check_polygons)
            start_idx = i + 1
            check_polygons = [ polygons_list[j] for j in range(i+1, polygon_count)
                               if is_two_bound_disjoint(polygon_boxes[i],polygon_boxes[j]) is False]
            adj_polygons = find_adjacent_polygons(polygons_list[i], check_polygons)
            # change polygon index to the entire polygons list
            adj_poly_idxs = [polygons_list.index(item) for item in adj_polygons ]

            # find index from the entire polygon list
            # adj_polygons, adj_poly_idxs = find_adjacent_polygons(polygons_list[i], polygons_list, Rtree=tree)

            # adj_polygons, adj_poly_idxs = find_adjacent_polygons(polygons_list[i], check_polygons, Rtree=tree)

            # find adjacent from entire list using tree, but slower
            # adjacent_polygons = [item for item in tree.query(polygons_list[i]) if item.intersection(polygons_list[i])]
            # adjacent_poly_idx = [polygons_list.index(item) for item in adjacent_polygons]
            # remove itself
            # adjacent_poly_idx.remove(i)
            # for idx in adjacent_poly_idx:
            #     ad_matrix[i, idx] = 1
            #     ad_matrix[idx, i] = 1  # also need the low part of matrix, or later polygon can not find previous neighbours

            # print(datetime.now(), '%d/%d'%(i, polygon_count),'cost', time.time() - t0)

            for j in adj_poly_idxs:
                ad_matrix[i, j] = 1
                ad_matrix[j, i] = 1  # also need the low part of matrix, or later polygon can not find previous neighbours
    elif process_num > 1:
        theadPool = Pool(process_num)
        parameters_list = [(i, polygons_list, polygon_boxes, i+1, polygon_count) for i in range(0,polygon_count)]
        results = theadPool.starmap(find_adjacent_polygons_from_sub, parameters_list)
        print(datetime.now(), 'finish parallel runing')
        for i, adj_polygons, adj_poly_idxs in results:
            # print(adj_poly_idxs)
            for j in adj_poly_idxs:
                ad_matrix[i, j] = 1
                ad_matrix[j, i] = 1  # also need the low part of matrix, or later polygon can not find previous neighbours
        # close it, to avoid error: OSError: [Errno 24] Too many open files
        theadPool.close()
    else:
        raise ValueError('wrong process_num: %d'%process_num)

    # print(ad_matrix)
    return ad_matrix


def get_surrounding_polygons(in_polygons,buffer_size):
    '''
    get polygons surround the input polygons
    similar to the one: "get_buffer_polygons" in "vector_features.py"
    Args:
        in_polygons:
        buffer_size:

    Returns: a list of expanding polygons

    '''
    # remove holes
    polygons = [ fill_holes_in_a_polygon(poly) for poly in  in_polygons]
    # buffer the polygons
    expansion_polygons = [ item.buffer(buffer_size) for item in polygons]
    # difference
    surround_polys = [exp_poly.difference(poly) for exp_poly, poly in zip(expansion_polygons,polygons)]

    return surround_polys

def merge_multi_geometries(geometry_list):
    # Merge geometries using unary_union
    merged_geometry = unary_union(GeometryCollection(geometry_list))

    # # Check if the result is already a single geometry
    # if isinstance(merged_geometry, GeometryCollection):
    #     return merged_geometry  # Return as is if it's a collection
    # else:
    #     return merged_geometry  # Return a single merged geometry
    # ouutside this function, to check if it's a single geometry or GeometryCollection
    return merged_geometry

def split_polygon_by_grids(polygon, grid_size_x=5000, grid_size_y=5000, min_grid_wh=500):
    """
    Splits a polygon into smaller grids of size grid_size_x × grid_size_y meters,
    and processes intersections, merging small cells on the edges into their direct neighbors.

    Parameters:
        polygon: The input Shapely polygon to split.
        grid_size_x: The width of each grid cell in meters (default is 5000m).
        grid_size_y: The height of each grid cell in meters (default is 5000m).
        min_grid_wh: Minimum width and height for a grid cell in meters (default is 500m).

    Returns:
        list: A list of resulting polygons after splitting and merging (Shapely Polygon objects).
    """
    # Get the bounds of the polygon (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = polygon.bounds

    # Step 1: Generate all grid boxes and assign them row/column numbers
    grid_cells = {}  # Dictionary to store grid cells by their (row, col) keys
    row = 0
    x = minx
    while x < maxx:
        col = 0
        y = miny
        while y < maxy:
            # Create a grid cell with given row and column
            grid_cells[(row, col)] = box(x, y, x + grid_size_x, y + grid_size_y)
            y += grid_size_y
            col += 1
        x += grid_size_x
        row += 1

    # Step 2: Calculate intersections and keep non empty intersections
    intersections = {}  # Store valid intersections by (row, col)
    for (row, col), grid_cell in grid_cells.items():
        # Calculate the intersection
        intersection = polygon.intersection(grid_cell)

        # Skip empty intersections
        if intersection.is_empty:
            continue
        intersections[(row, col)] = intersection

    # Step 3: Handle small cells by merging with neighbors
    results = {}  # Final list of polygons
    for (row, col), intersection in intersections.items():
        # Get the bounds of the intersection
        minx, miny, maxx, maxy = intersection.bounds
        width = maxx - minx
        height = maxy - miny

        # Check if the intersection is too small
        if width < min_grid_wh or height < min_grid_wh:
            # Merge with left or right neighbors first
            merged = False
            neighbors = [(row, col - 1), (row, col + 1), (row - 1, col), (row + 1, col)]  # Left, right, up, down neighbors
            for neighbor in neighbors:
                if neighbor not in intersections.keys():    # if outside the extent of all grid cells
                    continue
                
                neighbor_cell = intersections[neighbor]
                merged_cell = intersection.union(neighbor_cell).buffer(1)  # Use buffer to fix potential issues with invalid geometries
                # if isinstance(merged_cell, Polygon):
                #     merged = True
                    # merge the intersection with the neighbor
                results[neighbor] = merged_cell
                break
                    # basic.outputlogMessage("Merged a intersection to its neighbor")
                    # break
            
            # # If no neighbor is found to merge, abandon this intersection
            # if not merged:
            #     basic.outputlogMessage(f"Warning: Intersection at (row={row}, col={col}) is too small and cannot be merged with neighbors."
            #                            f" Abandoning it: {intersection}.")

        else:
            # If the intersection is valid and not too small, add it to the results
            results[(row, col)] = intersection
    
    # Step 4: If the intersection is not a single Polygon, keep the original grid cell
    for (row, col), intersection in results.items():
        if not isinstance(intersection, Polygon):
            basic.outputlogMessage(f"Warning: Intersection: {intersection} at (row={row}, col={col}) is not a single Polygon. Keeping the grid cell.")
            results[(row, col)] = grid_cells[(row, col)]


    return results.values()  # Return the list of resulting polygons


def merge_vector_files(file_list, save_path,format='ESRI Shapefile'):

    command_string = 'ogrmerge.py -o %s -f "%s" -single -progress '%(save_path,format)
    for item in file_list:
        command_string += ' "%s"'%item
    basic.os_system_exit_code(command_string)

def merge_vector_files_geopandas(file_list, save_path,format='ESRI Shapefile'):
    if os.path.isfile(save_path):
        print(f'{save_path} exists, skip merge_vector_files_geopandas')
        return
    # Read each shapefile into a GeoDataFrame and store in a list
    gdfs = [gpd.read_file(shp) for shp in file_list]
    # Concatenate all GeoDataFrames
    merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    # Save to a new shapefile
    merged_gdf.to_file(save_path,driver=format)
    basic.outputlogMessage(f'Saved to {save_path}')

def merge_shape_files(file_list, save_path):

    if os.path.isfile(save_path):
        print('%s already exists'%save_path)
        return True
    if len(file_list) < 1:
        raise IOError("no input shapefiles")

    ref_prj = map_projection.get_raster_or_vector_srs_info_proj4(file_list[0])

    # read polygons as shapely objects
    attribute_names = None
    polygons_list = []
    polygon_attributes_list = []

    b_get_field_name = False

    for idx, shp_path in enumerate(file_list):

        # check projection
        prj = map_projection.get_raster_or_vector_srs_info_proj4(file_list[idx])
        if prj != ref_prj:
            raise ValueError('Projection inconsistent: %s is different with the first one'%shp_path)

        shapefile = gpd.read_file(shp_path)
        if len(shapefile.geometry.values) < 1:
            basic.outputlogMessage('warning, %s is empty, skip'%shp_path)
            continue

        # go through each geometry
        for ri, row in shapefile.iterrows():
            # if idx == 0 and ri==0:
            if b_get_field_name is False:
                attribute_names = row.keys().to_list()
                attribute_names = attribute_names[:len(attribute_names) - 1]
                # basic.outputlogMessage("attribute names: "+ str(row.keys().to_list()))
                b_get_field_name = True

            polygons_list.append(row['geometry'])
            polygon_attributes = row[:len(row) - 1].to_list()
            if len(polygon_attributes) < len(attribute_names):
                polygon_attributes.extend([None]* (len(attribute_names) - len(polygon_attributes)))
            polygon_attributes_list.append(polygon_attributes)

    # save results
    save_polyons_attributes = {}
    for idx, attribute in enumerate(attribute_names):
        # print(idx, attribute)
        values = [item[idx] for item in polygon_attributes_list]
        save_polyons_attributes[attribute] = values

    save_polyons_attributes["Polygons"] = polygons_list
    polygon_df = pd.DataFrame(save_polyons_attributes)


    return save_polygons_to_files(polygon_df, 'Polygons', ref_prj, save_path)

def raster2shapefile(in_raster, out_shp=None,connect8=True, format='ESRI Shapefile'):
    # some time "ESRI Shapefile" may be failed is the raster is large and complex, it good to use "GPKG" (GeoPackage)
    if out_shp is None:
        if format.upper()=='GPKG':
            out_shp = os.path.splitext(in_raster)[0] + '.gpkg'
        else:
            out_shp = os.path.splitext(in_raster)[0] + '.shp'

    if os.path.isfile(out_shp):
        print('%s exists, skip'%out_shp)
        return out_shp

    commond_str = 'gdal_polygonize.py '
    # is Default is 4 connectedness.
    if connect8:
        commond_str += ' -8 '
    commond_str += in_raster + ' -b 1 '
    commond_str += ' -f  "%s" '%format + out_shp  # +  [layer] [fieldname]

    print(commond_str)
    res = os.system(commond_str)
    if res == 0:
        return out_shp
    else:
        return None


def points_to_LineString(point_list):
    # input a point in order, then output a line
    # for point in point_list:
    return LineString(point_list)


def line_segments_to_LineString(segment_list):
    # input a list of line segment: ((x1, y1), (x2, y2)),
    # then convert to LineString or (multi-linestring)

    line_2points_list = []
    for line in segment_list:
        p1, p2 = line
        line_2points_list.append(LineString([ [p1[0], p1[1]], [p2[0], p2[1]] ]))

    # combine them into a multi-linestring
    multi_line = MultiLineString(line_2points_list)
    # print(multi_line)  # prints MULTILINESTRING

    # now merge the lines
    # note that it will now merge only the contiguous portions into a component of a new multi-linestring
    merged_line = ops.linemerge(multi_line)
    # print(merged_line)
    # print(merged_line.geom_type)
    # line_list = list(merged_line)
    # print('line count:',len(line_list))
    # lenth_list = [ item.length for item in line_list]
    # print('max line length:', max(lenth_list) )

    return merged_line

def shapefile_to_ROIs_wkt(shp_path):
    polygons = read_shape_gpd_to_NewPrj(shp_path,'EPSG:4326')  # lat, lon
    if len(polygons) < 1:
        raise ValueError('No polygons in %s'%shp_path)
    ROIs_wkt = [str(p) for p in polygons ]
    return ROIs_wkt

def json_geometry_to_polygons(data_dict):
    return Polygon(data_dict['coordinates'][0])

def wkt_string_to_polygons(wkt_str):
    return shapely.wkt.loads(wkt_str)


def sample_points_within_polygon(polygon, b_grid=True, max_point_count=10):
    minx, miny, maxx, maxy = polygon.bounds
    grid_size = 5 if max_point_count < 5 else max_point_count
    if b_grid:
        # get grid points
        x_s = np.linspace(minx, maxx, num=grid_size)
        y_s = np.linspace(miny, maxy, num=grid_size)
    else:
        # get random points
        x_s = np.random.uniform(minx, maxx, grid_size)
        y_s = np.random.uniform(miny, maxy, grid_size)
        x_s = np.unique(x_s)
        y_s = np.unique(y_s)

    # to a grid
    x_grid, y_grid = np.meshgrid(x_s,y_s)

    point_list = [Point(x,y) for x,y in zip(x_grid.flatten(), y_grid.flatten())]

    gpd_points = gpd.GeoSeries(point_list)
    res = gpd_points.within(polygon)
    gpd_points_within = gpd_points[res].to_list()
    if len(gpd_points_within) > max_point_count:
        # gpd_points_within = random.choices(gpd_points_within,k=max_point_count)   # may have duplicate values
        gpd_points_within = random.sample(gpd_points_within,max_point_count)        # sample, get unique value
    return gpd_points_within
    # return point_list

def remove_polygon_boxes_touch_edge(in_vector, bounds, save_path=None, shrink_meters=10, b_save_removal=False):
    """
    Remove polygons/boxes that are not fully within the shrunken extent defined by `bounds`.
    The extent is shrunk inward by `shrink_meters` before checking. Prints info about removed features.

    Args:
        in_vector (str): Input vector file path (shapefile, .gpkg, etc.).
        bounds (tuple): (minx, miny, maxx, maxy) defining the extent.
        save_path (str or None): Output path. If None, overwrite input.
        shrink_meters (float): Amount to shrink the extent inward before checking.
    """
    if save_path is None:
        save_path = in_vector

    # Unpack bounds
    minx, miny, maxx, maxy = bounds
    extent_poly = box(minx, miny, maxx, maxy)
    shrunk_extent = extent_poly.buffer(-shrink_meters)

    if shrunk_extent.is_empty:
        print("Warning: Shrinking the extent by the specified meters produced an empty geometry. No features will be kept.")
        return None

    gdf = gpd.read_file(in_vector)

    # Keep only features that are fully within the shrunken extent
    within_shrunk = gdf.geometry.within(shrunk_extent)

    removed = gdf[~within_shrunk]
    kept = gdf[within_shrunk]

    # Reporting removed geometries
    if not removed.empty:
        print(f"Warning: removed {len(removed)} polygons/boxes close to defined extent in '{os.path.relpath(in_vector)}'")
        if b_save_removal:
            save_path_ext = os.path.splitext(save_path)
            save_removal = save_path_ext[0] + '_removal' + save_path_ext[1]
            removed.to_file(save_removal)

    if not kept.empty:
        kept.to_file(save_path)
        return save_path
    else:
        # if no geometry after removing
        print("No polygons/boxes remain after removal.")
        return None


def clip_geometries(input_path, save_path, mask, target_prj=None, format='ESRI Shapefile'):
    # If the mask is list-like with four elements (minx, miny, maxx, maxy), a faster rectangle clipping algorithm will be used
    # ref: https://geopandas.org/en/stable/docs/reference/api/geopandas.clip.html

    if isinstance(input_path,str) and os.path.isfile(input_path):
        shapefile = gpd.read_file(input_path)
    else:
        # if the input is already a geo dataframe
        shapefile = input_path
    if target_prj is not None:
        in_prj = map_projection.get_raster_or_vector_srs_info_proj4(input_path)
        if in_prj != target_prj:
            if version.parse(gpd.__version__) >= version.parse('0.7.0'):
                shapefile = shapefile.to_crs(target_prj)
            else:
                shapefile = shapefile.to_crs({'init': target_prj})
    shp_clip = gpd.clip(shapefile,mask)
    if shp_clip.empty:
        return None

    shp_clip.to_file(save_path, driver=format)
    return save_path

def clip_geometries_ogr2ogr(input_path, save_path, bounds, format='ESRI Shapefile'):
    # bounds: xmin, ymin, xmax, ymax
    # using ogr2ogr to crop vector file, if the vector file is very large (> 1GB),
    # we don't have to read all geometries to cpu memory, then use geopandas to clip it  (slow, out-of-memory)

    # noted: may need to check the projection before calling this function

    # ogr2ogr -progress -f GPKG -spat ${xmin} ${ymin} ${xmax} ${ymax}  ${dst} ${src}
    commond_str = 'ogr2ogr -f "%s"'%format      # -progress
    commond_str += ' -spat %s %s %s %s '%(str(bounds[0]), str(bounds[1]), str(bounds[2]), str(bounds[3]))
    commond_str += " %s %s"%(save_path,input_path)

    res = os.system(commond_str)
    if res == 0:
        return save_path
    else:
        return None

def polygons_overlap_another_group_in_file(polygon_list, projection, ref_shp, b_return_idx=False):
    # Create GeoDataFrame from the input polygon list
    group1 = gpd.GeoDataFrame({'geometry': polygon_list})
    group1.set_crs(epsg=projection, inplace=True)  # Set CRS from input projection
    # e.g.
    # group1.set_crs(epsg=4326, inplace=True)  # Set CRS to WGS84 (latitude/longitude)

    # Read the reference shapefile
    group2 = gpd.read_file(ref_shp)

    # Check and align CRS if needed
    if group1.crs != group2.crs:
        print('Warning: CRS in group 1 (%s) and group 2 (%s) is different. Re-projecting group 2 to %s.'
              % (group1.crs, group2.crs, group1.crs))
        group2 = group2.to_crs(group1.crs)

    # Perform spatial join to find overlapping or touching geometries
    overlap_touch = gpd.sjoin(group1, group2, how='inner', predicate='intersects')

    # Remove duplicate geometries in the result
    overlap_touch = overlap_touch.drop_duplicates(subset=['geometry'])

    if b_return_idx:
        # Return the indices of the overlapping polygons in the original list
        overlapping_indices = overlap_touch.index.to_list()
        return overlapping_indices

    # Return the overlapping geometries as a GeoDataFrame
    return overlap_touch

def geometries_overlap_another_group(input_shp, ref_shp, how='intersection'):
    # using geopandas to find geometries in "input_shp" that overlap any geometries in "ref_shp"
    # Read the shapefiles into GeoDataFrames
    group1 = gpd.read_file(input_shp)
    group2 = gpd.read_file(ref_shp)

    if group1.crs != group2.crs:
        print('Warning, CRS in group 1 (%s) and 2 (%s) is different, re-project group 2 to %s'%(group1.crs, group2.crs, group1.crs))
        group2 = group2.to_crs(group1.crs)

    # Perform overlay operation to find overlapping or touching geometries
    # overlap_touch = gpd.overlay(group1, group2, how=how)

    # Perform spatial join to find overlapping or touching geometries
    # overlap_touch = gpd.sjoin(group1, group2, how='inner', op='intersects')
    overlap_touch = gpd.sjoin(group1, group2, how='inner', predicate='intersects') # change to predicate for new geopandas, Jan 3, 2026

    # remove duplicated geometries in overlap_touch
    overlap_touch = overlap_touch.drop_duplicates(subset=['geometry'])    # only check geometry

    # save to disk for checking
    # overlap_touch.to_file(os.path.splitext(os.path.basename(ref_shp))[0] + '_sel.shp')

    return overlap_touch


def expand_polygon_to_specific_size(polygon, new_area_size):

    # check if polygons are in lat/lon (EPSG: 4326), need XY projection for calculating areas
    # https://math.stackexchange.com/questions/1889423/calculating-the-scale-factor-to-resize-a-polygon-to-a-specific-size
    # https://gis.stackexchange.com/questions/97963/how-to-surround-a-polygon-object-with-a-corridor-of-specified-width
    # https://stackoverflow.com/questions/72786576/how-to-scale-polygon-using-shapely

    # outside this function, need to make sure that the coordinate of polygons and unit of area_size is consistant
    # i.e., both are in meter or both are in degree

    if polygon.area >= new_area_size:
        return polygon
    # Calculate the scaling factor to achieve the desired area
    scaling_factor = (new_area_size / polygon.area) ** 0.5
    # Scale the polygon using the scaling factor
    expanded_polygon = polygon.buffer(0).scale(scaling_factor, scaling_factor)

    return expanded_polygon

 

def main(options, args):

    # ###############################################################
    # # test reading polygons with holes
    # polygons = read_polygons_gpd(args[0])
    # for idx, polygon in enumerate(polygons):
    #     if idx == 268:
    #         test = 1
    #         # print(polygon)
    #     print(idx, list(polygon.interiors))
    #     for inPoly in list(polygon.interiors):      # for holes
    #         print(inPoly)

    ###############################################################
    # test thinning a polygon
    # input_shp = args[0]
    # save_shp = args[1]
    # out_polygons_list = []
    # polygons = read_polygons_gpd(input_shp)
    # for idx, polygon in enumerate(polygons):
    #     # print(idx, polygon)
    #
    #     out_polygons = remove_narrow_parts_of_a_polygon(polygon,1.5)
    #     # save them
    #     out_polygons_list.append(out_polygons)
    #     print(idx)
    # import pandas as pd
    # import basic_src.map_projection as map_projection
    # out_polygon_df = pd.DataFrame({'out_Polygons': out_polygons_list
    #                             })
    #
    # wkt_string = map_projection.get_raster_or_vector_srs_info_wkt(input_shp)
    # save_polygons_to_files(out_polygon_df,'out_Polygons', wkt_string, save_shp)

    # remove_narrow_parts_of_polygons_shp(input_shp,save_shp, 1.5)

    ###############################################################
    # test reading polygons with
    input_shp = args[0]
    # read_polygons_gpd(input_shp)
    read_polygons_attributes_list(input_shp,'demD_mean')

    pass



if __name__=='__main__':
    usage = "usage: %prog [options] input_path output_file"
    parser = OptionParser(usage=usage, version="1.0 2016-10-26")
    parser.add_option("-p", "--para",
                      action="store", dest="para_file",
                      help="the parameters file")
    # parser.add_option("-s", "--used_file", action="store", dest="used_file",
    #                   help="the selectd used files,only need when you set --action=2")
    # parser.add_option('-o', "--output", action='store', dest="output",
    #                   help="the output file,only need when you set --action=2")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 2:
        parser.print_help()
        sys.exit(2)
    main(options, args)
