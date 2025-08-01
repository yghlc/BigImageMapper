#!/usr/bin/env python
# Filename: ARTS_to_classInt_polygons.py 
"""
introduction: convert the data from of ARTS (https://github.com/whrc/ARTS) to
shapefile only containing class_int

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 20 February, 2025
"""

import os,sys
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)

import pandas as pd

import datasets.vector_gpd as vector_gpd
import basic_src.basic as basic
import basic_src.map_projection as map_projection
import basic_src.io_function as io_function

from shapely.geometry import GeometryCollection
import math
from datetime import datetime

def merge_geometries(uID, geometry_list):
    if len(geometry_list) < 0:
        raise ValueError('No geometries')
    if len(geometry_list) == 1:
        return geometry_list[0]

    # merge multiple geometries
    merged_geometry = vector_gpd.merge_multi_geometries(geometry_list)
    if isinstance(merged_geometry, GeometryCollection):
        basic.outputlogMessage(f'Warning, the merging of overlap geometries for {uID} results in multiple geometry, '
                               f'try MultiPolygon_to_polygons')

        # for idx,item in enumerate(geometry_list):
        #     print('checking:', idx, item.geom_type, item)

        output = vector_gpd.MultiPolygon_to_polygons(0,merged_geometry)
        if len(output) == 1:
            basic.outputlogMessage('Converted to one geometry')
            return output[0]
        else:
            basic.outputlogMessage('Warning, There are still multiple geometries, still need to think how to handle this')

        # for values in output:
        #     print('testing:',values)

    return merged_geometry


def merge_geometry_with_the_same_UID(geometries,train_class,uID_list):
    if len(geometries) != len(train_class) or len(train_class) != len(uID_list):
        raise ValueError(f'The count are different: {len(geometries)}, {len(train_class)}, {len(uID_list)}')

    # Dictionary to store merged geometries for each unique UID
    merged_data = {}

    for geometry, t_class, uid in zip(geometries, train_class, uID_list):
        if uid not in merged_data:
            # If UID is not in the dictionary, initialize it
            merged_data[uid] = {"geometry": [geometry], "train_class": t_class}
        else:
            # If UID exists, merge the geometry and ensure classes match
            merged_data[uid]["geometry"].append(geometry)
            if merged_data[uid]["train_class"] != t_class:
                raise ValueError(f'Conflicting train classes for UID {uid}: {merged_data[uid]["train_class"]} and {t_class}')

    # merge geometry if there are multiple
    merged_geometries = [merge_geometries(uid, merged_data[uid]["geometry"]) for uid in merged_data.keys()]

    # Convert merged data back into lists
    merged_train_classes = [v["train_class"] for v in merged_data.values()]
    merged_uIDs = list(merged_data.keys())

    return merged_geometries, merged_train_classes, merged_uIDs

def save_to_a_gpkg_file(save_attributes, wkt_string, output):
    polygon_df = pd.DataFrame(save_attributes)
    vector_gpd.save_polygons_to_files(polygon_df, 'Polygons', wkt_string, output, format='GPKG')
    print(datetime.now(), f'saved to {output}')

def save_original_geometries(geometries, train_class_int, id_list, uIDs, wkt_string, save_path):
    geometry_type = [ item.geom_type for item in geometries]
    save_attributes = {'Polygons': geometries, 'id': id_list, 'UID': uIDs, 'geom_type':geometry_type, 'class_int': train_class_int}
    save_to_a_gpkg_file(save_attributes, wkt_string, save_path)


def save_bounding_boxes(geometries, train_class_int, id_list, uIDs, wkt_string, save_path):

    geometry_type = [item.geom_type for item in geometries]
    bounding_boxes = [vector_gpd.convert_bounds_to_polygon(vector_gpd.get_polygon_bounding_box(item)) if type=='Polygon' and c_int==1 else item
                      for item, type,c_int in zip(geometries, geometry_type, train_class_int) ]

    b_bound_box = [ 'Yes' if item=='Polygon' and c_int==1 else 'No' for item, c_int in zip(geometry_type, train_class_int)]
    save_attributes = {'Polygons': bounding_boxes, 'id': id_list, 'UID': uIDs, 'geom_type':geometry_type,
                       'IsBox':b_bound_box, 'class_int': train_class_int}

    save_to_a_gpkg_file(save_attributes, wkt_string, save_path)

def convert_MultiPolygon_to_polygons(geometries_list, train_class):
    output_list = []
    for idx, geom in enumerate(geometries_list):
        if geom.geom_type == "MultiPolygon" and train_class[idx]=="Positive":
            polygon_list = vector_gpd.MultiPolygon_to_polygons(idx, geom)
            if len(polygon_list) == 1:
                output_list.append(polygon_list[0])
            else:
                # raise ValueError(f'error: idx: {idx}, {len(polygon_list)} polygons')
                area_list = [item.area for item in polygon_list ]
                max_idx = area_list.index(max(area_list))
                area_list_sorted = sorted(area_list, reverse=True)
                basic.outputlogMessage(f'Waning, fid: {idx+1} is a MultiPolygon, {len(polygon_list)} polygons, keep the maximum one, their areas: {area_list_sorted}')
                # print(polygon_list[max_idx].area)
                output_list.append(polygon_list[max_idx])
        else:
            output_list.append(geom)

    return output_list

def group_overlap_circles(geom_circle_list,overlap_thr, max_group_area):

    group_list = vector_gpd.group_overlap_polygons(geom_circle_list,overlap_threshold=overlap_thr,
                                                   group_max_area=max_group_area, b_verbose=False)

    print(f'Overlap grouping: group {len(geom_circle_list)} polygons into {len(group_list)} groups')
    # print(group_list[0])
    # print(group_list[1])
    # print(group_list[2])
    # print(group_list[3])
    # group_ids = [item+1  for item in range(len(group_list))]
    return group_list


def save_group_merged_polygons(group_list, geom_circle_list,id_list, train_class_int, wkt_string, output, poly_max_w=3000, poly_max_height=3000):

    merged_circles = []
    merged_id_list = []
    train_class_list = []
    b_class_all_same_list = []
    class_int_list = []
    group_id_list = []

    for idx, a_group in enumerate(group_list):
        circles = [geom_circle_list[item] for item in a_group]

        merged_id_list.append( ','.join([ str(id_list[item]) for item in a_group  ]) )
        sub_train_class_int = [train_class_int[item] for item in a_group]
        train_class_list.append( ','.join([ str(item) for item in sub_train_class_int]) )
        group_id_list.append(idx+1)

        unique_classes = list(set(sub_train_class_int))
        if len(unique_classes) == 1:
            b_class_all_same_list.append('Yes')
            class_int_list.append(unique_classes[0])
        else:
            b_class_all_same_list.append('No')
            class_int_list.append(-1)

        a_merged_circle = vector_gpd.merge_multi_geometries(circles)
        ### if the merged circle is too large, then split it by grids ###
        # Get the bounds of the clipped grid cell
        poly_minx, poly_miny, poly_maxx, poly_maxy = a_merged_circle.bounds
        poly_width = poly_maxx - poly_minx
        poly_height = poly_maxy - poly_miny
        if poly_width > poly_max_w or poly_height > poly_max_height:
            split_polygons = vector_gpd.split_polygon_by_grids(a_merged_circle,poly_max_w,poly_max_height,min_grid_wh=200)
            for s_i, s_poly in enumerate(split_polygons):
                merged_circles.append(s_poly)
                # adding attributes: attribute and polygons should have the same length
                if s_i > 0:
                    merged_id_list.append('')
                    train_class_list.append('')
                    b_class_all_same_list.append('')
                    group_id_list.append(group_id_list[-1])
                    class_int_list.append(class_int_list[-1])

        else:
            merged_circles.append(a_merged_circle)

    u_id_list = [ idx+1 for idx in range(len(merged_circles))]

    save_attributes = {'Polygons': merged_circles, 'uid':u_id_list ,'gid': group_id_list, 'merged_id':merged_id_list, 'merged_class':train_class_list,
                       'same_class':b_class_all_same_list, 'class_int':class_int_list}
    save_to_a_gpkg_file(save_attributes, wkt_string, output)



def ARTS_to_classInt_polygons(input,output,buff_radius=500):
    # read the existing data, convert to points
    geometries, train_class = vector_gpd.read_polygons_attributes_list(input, 'TrainClass',
                                                                       b_fix_invalid_polygon=True)
    print(f'read {len(geometries)} features')
    uID_list = vector_gpd.read_attribute_values_list(input,'UID')

    # testing
    # idx_query = uID_list.index('24575e45-044f-5927-81af-63d0b3239dd6')
    # sel_geometry = geometries[idx_query]
    # print(sel_geometry)
    # print('geom_type', sel_geometry.geom_type)
    # sys.exit(0)
    geometries = convert_MultiPolygon_to_polygons(geometries, train_class)
    # sys.exit(0)

    # merge polygons with the same UID
    m_geometries, m_train_classes, m_uIDs = merge_geometry_with_the_same_UID(geometries,train_class, uID_list)

    train_class_int = [1 if tclass == 'Positive' else 0 for tclass in m_train_classes ]
    geom_centrioid_list = [vector_gpd.get_polygon_centroid(item) for item in m_geometries]

    geom_circle_list = [item.buffer(buff_radius) for item in geom_centrioid_list]

    id_list = [idx+1 for idx in range(len(train_class_int)) ]

    save_polyons_attributes = {'Polygons':geom_circle_list, 'id': id_list, 'UID':m_uIDs, 'class_int':train_class_int}
    wkt_string = map_projection.get_raster_or_vector_srs_info_proj4(input)
    save_to_a_gpkg_file(save_polyons_attributes,wkt_string, output)


    # group overlap circles and merged them, for slilitating
    # overla_thr = 0.3*math.pi * (buff_radius ** 2)  # if more than 30% overlap, group them
    # max_group_area = 3*math.pi * (buff_radius ** 2)

    # group all connecting circles, if they are too large, split them by grids
    overla_thr = 0.1 
    max_group_area = 3000*math.pi * (buff_radius ** 2)

    poly_max_width = 2000
    poly_max_height = 2000
    group_list = group_overlap_circles(geom_circle_list,overla_thr, max_group_area)
    output_group = io_function.get_name_by_adding_tail(output,'groupMerge')
    save_group_merged_polygons(group_list,geom_circle_list, id_list, train_class_int,wkt_string,output_group,
                               poly_max_w=poly_max_width, poly_max_height=poly_max_height)


    # save positive and negative circles separately
    output_class_1 = io_function.get_name_by_adding_tail(output,'c1')
    vector_gpd.save_shapefile_subset_as_valueInlist(output,output_class_1,'class_int',[1],format='GPKG')
    output_class_0 = io_function.get_name_by_adding_tail(output, 'c0')
    vector_gpd.save_shapefile_subset_as_valueInlist(output, output_class_0, 'class_int', [0], format='GPKG')

    # save original polygons or points
    output_orginal_polys = io_function.get_name_by_adding_tail(output,'org')
    save_original_geometries(m_geometries,train_class_int,id_list,m_uIDs,wkt_string,output_orginal_polys)
    output_orginal_polys_c1 = io_function.get_name_by_adding_tail(output_orginal_polys,'c1')
    vector_gpd.save_shapefile_subset_as_valueInlist(output_orginal_polys, output_orginal_polys_c1, 'class_int', [1], format='GPKG')

    # save bounding boxes for original polygons
    output_bounding_box = io_function.get_name_by_adding_tail(output,'box')
    save_bounding_boxes(m_geometries, train_class_int, id_list, m_uIDs, wkt_string, output_bounding_box)
    output_bounding_box_c1 = io_function.get_name_by_adding_tail(output_bounding_box,'c1')
    vector_gpd.save_shapefile_subset_as_valueInlist(output_bounding_box, output_bounding_box_c1, 'class_int', [1],format='GPKG')


def main(options, args):
    input = args[0]
    output = options.save_path
    if output is None:
        output = io_function.get_name_by_adding_tail(input,'ClassInt')
    ARTS_to_classInt_polygons(input,output)
    pass


if __name__ == '__main__':
    usage = "usage: %prog [options] random1.shp "
    parser = OptionParser(usage=usage, version="1.0 2025-02-20")
    parser.description = 'Introduction: convert dataformat from ARTS to simpley polygons'

    parser.add_option("-s", "--save_path",
                      action="store", dest="save_path",
                      help="the file path for saving the results")


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)
