#!/usr/bin/env python
# Filename: validate_random_samples.py 
"""
introduction: pre-processing and post-processing of random samples

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 20 January, 2025
"""

import os,sys
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
# import parameters
import basic_src.basic as basic
from datetime import datetime
import basic_src.io_function as io_function
import basic_src.map_projection as map_projection
import datasets.vector_gpd as vector_gpd
import shapely
from shapely.strtree import STRtree

import geopandas as gpd
import pandas as pd
import math

def merge_shapefiles_remove_duplicates(shapefiles, output_path, unique_column='polyID'):
    """
    Merges multiple shapefiles and removes redundant records based on a unique column.

    Parameters:
        shapefiles (list): List of file paths to the shapefiles.
        output_path (str): Path to save the merged shapefile.
        unique_column (str): The column to check for uniqueness (default: 'polyID').
    """
    # Load and concatenate all shapefiles into a single GeoDataFrame
    gdf_list = [gpd.read_file(shp) for shp in shapefiles]
    merged_gdf = pd.concat(gdf_list, ignore_index=True)

    total_records = len(merged_gdf)

    # Drop duplicate rows based on the unique_column
    unique_gdf = merged_gdf.drop_duplicates(subset=unique_column)
    unique_records = len(unique_gdf)
    removed_records = total_records - unique_records

    # Save the result to a new shapefile
    unique_gdf.to_file(output_path)

    basic.outputlogMessage(f"Merged shapefiles and removed duplicated records (count: {removed_records}), saved to: {output_path}")
    return output_path


def split_shapefile(input_shp, count_per_group=200, output_dir='./'):
    """
    Splits a GeoDataFrame into groups with roughly equal sizes and saves each group as a shapefile.
    """
    # Load the input shapefile into a GeoDataFrame
    gdf = gpd.read_file(input_shp)
    total_count = len(gdf)

    # Calculate the number of groups needed
    num_groups = math.ceil(total_count / count_per_group)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    base_name = io_function.get_name_no_ext(input_shp)
    save_file_list = []

    # Split the GeoDataFrame into groups and save each group as a shapefile
    for i in range(num_groups):
        start_idx = i * count_per_group
        end_idx = start_idx + count_per_group
        group_gdf = gdf.iloc[start_idx:end_idx]

        # Generate the output filepath
        output_path = os.path.join(output_dir, base_name+f"_g{i + 1}_{len(group_gdf)}s.shp")

        # Save the group to a shapefile
        group_gdf.to_file(output_path)
        basic.outputlogMessage(f"Saved group {i + 1} with {len(group_gdf)} features to: {output_path}")
        save_file_list.append(output_path)

    basic.outputlogMessage(f"Splitting complete! Total groups created: {num_groups}")
    return save_file_list


def get_unique_sample_for_validation(shp_list, save_path, unique_column='polyID', count_per_group=200):

    merge_shapefiles_remove_duplicates(shp_list,save_path,unique_column=unique_column)

    save_dir = io_function.get_name_no_ext(save_path) + '_groups'
    split_shp_list = split_shapefile(save_path,count_per_group,save_dir)
    return split_shp_list

def validate_against_existing_results(existing_data, in_shp_list, radius=500):
    '''
    validate the results in shapefile against some exist results, if the location match, set it as Yes, otherwise, do nothing
    if the location match within 2*radius, then set it as true
    :param existing_data:
    :param in_shp_list:
    :param radius:
    :return:
    '''

    # read the existing data, convert to points
    geometries, train_class = vector_gpd.read_polygons_attributes_list(existing_data,'TrainClass',b_fix_invalid_polygon=False)
    print(f'read {len(geometries)} features')
    geometries_pos = [item for item, tclass in zip(geometries,train_class) if tclass=='Positive']
    print(f'read {len(geometries_pos)} positive features')
    # print(geometries_pos[0])
    geom_pos_centrioid_list = [ vector_gpd.get_polygon_centroid(item) for item in geometries_pos]
    print(f'read {len(geom_pos_centrioid_list)} point features')
    geom_pos_circle_list = [item.buffer(radius) for item in geom_pos_centrioid_list]

    # exist_prj = map_projection.get_raster_or_vector_srs_info_proj4(existing_data)
    exist_prj = map_projection.get_raster_or_vector_srs_info_epsg(existing_data)
    # print(exist_prj)

    ## save for checking
    # save_circle_dict = {'id': [item+1 for item in range(len(geom_pos_circle_list))], "Polygon": geom_pos_circle_list}
    # save_pd = pd.DataFrame(save_circle_dict)
    # ref_prj = map_projection.get_raster_or_vector_srs_info_proj4(existing_data)
    # vector_gpd.save_polygons_to_files(save_pd, 'Polygon', ref_prj, 'geom_circles.shp')

    # build a tree
    tree = STRtree(geom_pos_circle_list)

    # validate input shapefile list
    for idx, in_shp in enumerate(in_shp_list):
        validate_count = 0
        already_count = 0
        print(datetime.now(), f'({idx + 1}/{len(in_shp_list)})validate points in {os.path.basename(in_shp)}')

        # check the projection
        in_shp_prj = map_projection.get_raster_or_vector_srs_info_epsg(in_shp)
        # print(in_shp_prj)

        if exist_prj != in_shp_prj:
            # raise ValueError(f'the map projection is different between {existing_data} and {in_shp_prj}')
            basic.outputlogMessage(f'Warning, repoject the shapefile {os.path.basename(in_shp)} to {exist_prj}')
            in_geometries = vector_gpd.read_shape_gpd_to_NewPrj(in_shp,exist_prj)
            validates = vector_gpd.read_attribute_values_list(in_shp, 'validate')
        else:
            in_geometries, validates = vector_gpd.read_polygons_attributes_list(in_shp, 'validate',
                                                                                b_fix_invalid_polygon=False)

        # check one by one
        for p_idx, (geom, check) in enumerate(zip(in_geometries,validates)):
            # if validate results already there, then skip
            if check is not None:
                already_count += 1
                continue
            # inter_or_touch_list = vector_gpd.find_adjacent_polygons(geom,geom_pos_circle_list,Rtree=tree)
            # print(geom)
            # print(geom_pos_circle_list[0])
            inter_or_touch_list =  tree.query(geom)
            # print(inter_or_touch_list)
            if len(inter_or_touch_list) > 0:
                validates[p_idx] = 'Yes-auto'
                validate_count += 1

        print(f'validate {validate_count}+{already_count},(total: {len(in_geometries)}) points in {os.path.basename(in_shp)}')


        # save the results into the shapefile
        vector_gpd.add_attributes_to_shp(in_shp,{'validate':validates})



def test_validate_against_existing_results():
    existing_data = os.path.expanduser('~/Data/published_data/Yili_Yang_etal_ARTS_The-Arctic-Retrogressive-Thaw-Slumps-Data-Set/'
                                       'ARTS_main_dataset_v.2.1.0.gpkg')

    shp_dir = os.path.expanduser('~/Data/slump_demdiff_classify/clip_classify/merge_classify_result_v2/classID1_occur7_012110_Sel_merge_groups')
    group_shp_list = io_function.get_file_list_by_pattern(shp_dir, '*.shp')

    print(datetime.now(), 'existing data:', existing_data)
    print(datetime.now(), 'group_shp_list:')
    [print(item) for item in group_shp_list]

    validate_against_existing_results(existing_data, group_shp_list)


def main(options, args):
    res_shp_list = args
    res_shp_list = [os.path.abspath(item) for item in res_shp_list]
    save_path = options.save_path
    count_each_group = options.count_per_group
    existing_data = options.existing_data

    if save_path is not None:
        # pre-processing task, remove duplciates, and split them into different groups
        shp_file_list = get_unique_sample_for_validation(res_shp_list, save_path, count_per_group=count_each_group)

        if existing_data is not None:
            validate_against_existing_results(existing_data, shp_file_list)

    else:
        # post-processing, copy the validated result to original shapefiles
        # to add
        pass

    pass


if __name__ == '__main__':
    # test_validate_against_existing_results()
    # sys.exit(0)

    usage = "usage: %prog [options] random1.shp random2.shp random3.shp ... "
    parser = OptionParser(usage=usage, version="1.0 2025-01-20")
    parser.description = 'Introduction: pre-processing and post-processing of random samples for validation'

    parser.add_option("-s", "--save_path",
                      action="store", dest="save_path",
                      help="the file path for saving the results")

    parser.add_option("-e", "--existing_data",
                      action="store", dest="existing_data",
                      help="the file path of existing data")

    parser.add_option("-c", "--count_per_group",
                      action="store", dest="count_per_group", type=int, default=200,
                      help="the sample count for each group")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)

