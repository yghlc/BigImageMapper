#!/usr/bin/env python
# Filename: validate_random_samples.py 
"""
introduction: pre-processing and post-processign of random samples

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

    basic.outputlogMessage(f"Splitting complete! Total groups created: {num_groups}")


def get_unique_sample_for_validation(shp_list, save_path, unique_column='polyID', count_per_group=200):

    merge_shapefiles_remove_duplicates(shp_list,save_path,unique_column=unique_column)

    save_dir = io_function.get_name_no_ext(save_path) + '_groups'
    split_shapefile(save_path,count_per_group,save_dir)


def main(options, args):
    res_shp_list = args
    res_shp_list = [os.path.abspath(item) for item in res_shp_list]
    save_path = options.save_path
    count_each_group = options.count_per_group

    if save_path is not None:
        # pre-processing task, remove duplciates, and split them into different groups
        get_unique_sample_for_validation(res_shp_list, save_path, count_per_group=count_each_group)
    else:
        # post-processing, copy the validated result to original shapfiles
        # to add
        pass

    pass


if __name__ == '__main__':

    usage = "usage: %prog [options] random1.shp random2.shp random3.shp ... "
    parser = OptionParser(usage=usage, version="1.0 2025-01-20")
    parser.description = 'Introduction: pre-processing and post-processing of random samples for validation'

    parser.add_option("-s", "--save_path",
                      action="store", dest="save_path",
                      help="the file path for saving the results")

    parser.add_option("-c", "--count_per_group",
                      action="store", dest="count_per_group", type=int, default=200,
                      help="the sample count for each group")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)

