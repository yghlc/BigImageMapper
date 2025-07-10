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
    print(f'saved to {output}')

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
