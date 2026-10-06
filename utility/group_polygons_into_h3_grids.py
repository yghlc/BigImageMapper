#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Introduction: Dividing a set of polygons into groups based on H3 grids. 
If other grid systems is provides (reference_grids_shp), the polygons will be grouped based on the reference grids.


Author: Huang Lingcao
Email: huanglingcao@gmail.com
Created: 2026-10-06
"""

import os, sys
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)


import basic_src.basic as basic
import basic_src.io_function as io_function
import datasets.vector_gpd as vector_gpd
import datasets.geo_index_h3 as geo_index_h3

import geopandas as gpd
import pandas as pd


def group_polygons_into_h3_grids(input_vector, save_dir, min_count_per_group=200, max_count_per_group=500, ref_h3_resolution=3):

    # get centroid of each polygon and convert them to lat/lon, then get the H3 ID of each polygon, and group them based on H3 IDs
    if os.path.isdir(save_dir) is False:
        io_function.mkdir(save_dir)

    gdf = gpd.read_file(input_vector)
    original_crs = gdf.crs

    # ------------------------------------------------------------------
    # Compute centroids, then convert centroids to EPSG:4326
    # ------------------------------------------------------------------
    if original_crs.is_geographic:
        # Use a projected CRS for centroid calculation
        centroid_gdf = gdf.to_crs(epsg=3413).copy()
        centroid_gdf["geometry"] = centroid_gdf.geometry.centroid
        centroid_gdf = centroid_gdf.to_crs(epsg=4326)
    else:
        centroid_gdf = gdf.copy()
        centroid_gdf["geometry"] = centroid_gdf.geometry.centroid
        centroid_gdf = centroid_gdf.to_crs(epsg=4326)

    gdf["_centroid_lon"] = centroid_gdf.geometry.x.values
    gdf["_centroid_lat"] = centroid_gdf.geometry.y.values

    # ------------------------------------------------------------------
    # Iterative grouping using while loop
    # ------------------------------------------------------------------
    final_groups = []
    groups_to_process = [(gdf, ref_h3_resolution)]

    while groups_to_process:
        current_group, resolution = groups_to_process.pop(0)
        current_group = current_group.copy()

        current_group["_h3_resolution"] = resolution
        current_group["h3_id"] = current_group.apply(
            lambda row: geo_index_h3.get_h3_cell_id(
                row["_centroid_lat"],
                row["_centroid_lon"],
                resolution
            ),axis=1 )

        for h3_id, subgroup in current_group.groupby("h3_id"):
            subgroup = subgroup.copy()

            if len(subgroup) <= max_count_per_group:
                final_groups.append(subgroup)
            else:
                if resolution < 15:
                    # Increase H3 resolution and process this subgroup again
                    groups_to_process.append((subgroup, resolution + 1))
                else:
                    # H3 resolution cannot go beyond 15.
                    # Keep this group even though it exceeds max_count_per_group.
                    final_groups.append(subgroup)

    # merge small groups into larger groups if they are smaller than min_count_per_group
    small_groups = [group for group in final_groups if len(group) < min_count_per_group]
    large_groups = [group for group in final_groups if len(group) >= min_count_per_group]
    merged_groups = []
    for group in small_groups:
        if merged_groups and len(merged_groups[-1]) + len(group) <= max_count_per_group:
            merged_groups[-1] = pd.concat([merged_groups[-1], group])
        else:
            merged_groups.append(group)

    merged_groups.extend(large_groups)

    # ------------------------------------------------------------------
    # Save final groups
    # ------------------------------------------------------------------
    output_files = []

    for idx, group in enumerate(merged_groups, start=1):
        group = group.copy()
        h3_id = group["h3_id"].iloc[0]
        resolution = group["_h3_resolution"].iloc[0]
        
        filename = f"h3_res{resolution}_{h3_id}_g{idx+1}_{len(group)}s.gpkg"
        output_path = os.path.join(save_dir, filename)
        if os.path.isfile(output_path):
            print(f"Warning: {output_path} already exists, skip, remove it if needed.")
            continue

        # Restore original CRS
        group = gpd.GeoDataFrame(group, geometry="geometry", crs=original_crs)
        # Remove helper columns from output if desired
        group = group.drop( columns=["_centroid_lon", "_centroid_lat"], errors="ignore")


        group.to_file(output_path, driver="GPKG", layer="polygons")
        print(f"Saved group {idx+1} with {len(group)} polygons to {output_path}")
        output_files.append(str(output_path))

    print(f'save {len(output_files)} files into {save_dir}')
    
    return output_files


def main(options, args):
    input_vector = args[0]
    save_dir = args[1]
    b_h3_grid = options.b_h3_grid # to h3 grid in filename, and will use it
    reference_grids_shp = options.reference_grids_shp # if not None, to use reference grid
    min_count_per_group = options.min_count_per_group
    max_count_per_group = options.max_count_per_group
    ref_h3_resolution = options.ref_h3_resolution

    if b_h3_grid:
        
        save_dir = os.path.join(save_dir, io_function.get_name_no_ext(input_vector))
        group_polygons_into_h3_grids(input_vector, save_dir, min_count_per_group=min_count_per_group, max_count_per_group=max_count_per_group, 
                                     ref_h3_resolution=ref_h3_resolution)




if __name__ == "__main__":
    usage = "usage: %prog [options] input_vector save_dir "
    parser = OptionParser(usage=usage, version="1.0 2026-10-06")
    parser.description = 'Introduction: divide a set of polygons into groups based on H3 grids'

    parser.add_option("-r", "--reference_grids_shp",
                      action="store", dest="reference_grids_shp",
                      help="the vector file of reference grids")

    parser.add_option("-g", "--b_h3_grid",
                    action="store_true", dest="b_h3_grid", default=False,
                    help="if set, it means H3 IDs are in filenames, and the H3 grid system will be used")
    
    parser.add_option("-l", "--ref_h3_resolution",
                    action="store", dest="ref_h3_resolution", type=int, default=3,
                    help="the reference resolution of H3 grids")

    parser.add_option("-n", "--min_count_per_group",
                    action="store", dest="min_count_per_group", type=int, default=200,
                    help="the minimum number of polygons per group")

    parser.add_option("-m", "--max_count_per_group",
                action="store", dest="max_count_per_group", type=int, default=500,
                help="the maximum number of polygons per group, if the number of polygons in a group is larger than this value, " \
                "it will be divided into sub-groups using Child H3 grids, "
                "and the resolution of child H3 grids will be increased by 1 each time until the number of polygons in a group is smaller than this value")
    

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) != 2:
        parser.print_help()
        sys.exit(2)
    main(options, args)