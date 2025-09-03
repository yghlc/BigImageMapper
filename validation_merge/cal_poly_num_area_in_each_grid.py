#!/usr/bin/env python
# Filename: cal_poly_num_area_in_each_grid.py
"""
introduction: calculate the number and areas of different mapping results (polygon) within or intersect grids or cells


authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 02 September, 2025
"""

import os,sys
import time
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)

import basic_src.io_function as io_function
import basic_src.basic as basic
import datasets.vector_gpd as vector_gpd

from datetime import datetime

# import dask_geopandas as dgpd
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np

def check_input_vector_files(grid_vector,in_poly_vector,column_pre_name,save_path):
    io_function.is_file_exist(grid_vector)
    io_function.is_file_exist(in_poly_vector)

    column_count = f"{column_pre_name}_C"
    column_area = f"{column_pre_name}_A"

    if os.path.isfile(f"{column_count}.npy"):
        print(f'{column_count}.npy already exist, skip calculating')
        return None, None

    if os.path.isfile(save_path):
        if vector_gpd.is_field_name_in_shp(save_path,column_count):
            print(f'Column {column_count} already in {save_path}, skip calculating')
            return None, None

    grid_prj = vector_gpd.get_projection(grid_vector)
    in_poly_proj = vector_gpd.get_projection(in_poly_vector)
    if grid_prj != in_poly_proj:
        raise ValueError(f'The map projection between {grid_vector} and {in_poly_vector} is different')

    if grid_prj == "EPSG:4326":
        raise ValueError('Not support lat/long projection')

    return column_count, column_area

def calculate_poly_count_area_in_each_grid(grid_vector, in_poly_vector, save_path, column_pre_name='poly',
                                           b_poly_bounds = False):

    column_count, column_area = check_input_vector_files(grid_vector,in_poly_vector,column_pre_name,save_path)
    if column_count is None:
        return

    grid_gpd = gpd.read_file(grid_vector,engine="pyogrio")
    poly_gpd = gpd.read_file(in_poly_vector,engine="pyogrio")
    if b_poly_bounds:
        poly_gpd['geometry'] = poly_gpd.bounds.apply(lambda row: box(row.minx, row.miny, row.maxx, row.maxy), axis=1)


    # Spatial join: for each grid, find intersecting polygons
    joined = gpd.sjoin(grid_gpd, poly_gpd, how="left", predicate="intersects")
    # print(joined)

    # For area calculation, get the intersection geometries
    joined["intersection"] = joined.apply(
        lambda row: row.geometry.intersection(poly_gpd.loc[row.index_right].geometry)
        if pd.notnull(row.index_right) else None, axis=1)


    joined["intersect_area"] = joined["intersection"].area
    # print(joined)

    # Group by grid cell index and aggregate
    grid_summary = (joined.groupby(joined.index).agg(poly_count=("index_right", lambda x: x.notnull().sum()),
             total_intersect_area=("intersect_area", "sum")))

    # Merge results back into grid
    grid_result = grid_gpd.copy()
    grid_result[column_count] = grid_summary["poly_count"]
    grid_result[column_area] = grid_summary["total_intersect_area"]

    # Fill NaNs with 0 (for grid cells with no intersections)
    grid_result = grid_result.fillna({column_count: 0, column_area: 0})

    # Save to desired path
    grid_result.to_file(save_path)
    print(datetime.now(),f'saved to {save_path}')


# def calculate_poly_count_area_in_each_grid_dask(grid_vector, in_poly_vector, save_path, column_pre_name='poly',
#                                                 b_poly_bounds=False, npartitions=32):
#     """
#     Parallel calculation of polygon count and intersection area in each grid cell using Dask-GeoPandas.
#     """
#
#     # script write by AI (this is GPT-5 version), also try GPT-4.1 version (output some duplidated result), but not working
#     print('This version is not working')
#
#     column_count, column_area = check_input_vector_files(grid_vector,in_poly_vector,column_pre_name,save_path)
#     if column_count is None:
#         return
#
#     # Read data
#     grid_gdf = gpd.read_file(grid_vector)
#
#     # Ensure a stable grid_id for later merge
#     if 'grid_id' not in grid_gdf.columns:
#         grid_gdf = grid_gdf.reset_index(drop=False).rename(columns={'index': 'grid_id'})
#
#     poly_gdf = gpd.read_file(in_poly_vector)
#
#     # Optional: replace polygon geometries with their bounds (version-compatible)
#     if b_poly_bounds:
#         poly_gdf['geometry'] = poly_gdf.bounds.apply(lambda row: box(row.minx, row.miny, row.maxx, row.maxy), axis=1)
#
#     # Add a stable polygon ID
#     if 'poly_id' not in poly_gdf.columns:
#         poly_gdf = poly_gdf.reset_index(drop=False).rename(columns={'index': 'poly_id'})
#
#     # Dask GeoDataFrames
#     dgrid = dgpd.from_geopandas(grid_gdf, npartitions=npartitions)
#
#     # Keep only necessary columns on the right side to minimize payload
#     right_geom_col = poly_gdf.geometry.name
#     dpoly = dgpd.from_geopandas(poly_gdf[['poly_id', right_geom_col]], npartitions=max(1, npartitions // 4))
#
#     # Spatial join (left geometry retained). Ensure only necessary columns are kept post-join.
#     joined = dgpd.sjoin(dgrid, dpoly, how="inner", predicate="intersects")
#
#     # Prepare a lightweight pandas lookup for right geometries to merge inside partitions
#     right_lookup = poly_gdf[['poly_id', right_geom_col]].rename(columns={right_geom_col: 'poly_geometry'}).copy()
#
#     # Define meta precisely to avoid mismatches. We will return only grid_id, poly_id, intersect_area, geometry.
#     meta_df = gpd.GeoDataFrame(
#         {
#             'grid_id': pd.Series(dtype=grid_gdf['grid_id'].dtype if 'grid_id' in grid_gdf else 'int64'),
#             'poly_id': pd.Series(dtype=right_lookup['poly_id'].dtype),
#             'intersect_area': pd.Series(dtype='float64'),
#         },
#         geometry=gpd.array.GeometryDtype()
#     )
#
#     def add_intersection_area(df, right_lookup):
#         # df is a pandas GeoDataFrame partition from the joined result.
#         if len(df) == 0:
#             # Return empty GeoDataFrame with expected columns and geometry dtype
#             return gpd.GeoDataFrame({'grid_id': pd.Series([], dtype=df['grid_id'].dtype if 'grid_id' in df else 'int64'),
#                                      'poly_id': pd.Series([], dtype=right_lookup['poly_id'].dtype),
#                                      'intersect_area': pd.Series([], dtype='float64')},
#                                     geometry=gpd.GeoSeries([], crs=getattr(df, 'crs', None)))
#
#         # Keep only needed columns to avoid carrying unexpected ones
#         cols_needed = ['grid_id', 'poly_id', df.geometry.name]
#         cols_needed = [c for c in cols_needed if c in df.columns]
#         sub = df[cols_needed].copy()
#
#         # Merge to obtain right geometry
#         sub = sub.merge(right_lookup, on='poly_id', how='left', copy=False, validate='m:1')
#
#         # Compute intersection area; handle nulls
#         left_geom = sub.geometry
#         right_geom = sub['poly_geometry']
#         inter = left_geom.intersection(right_geom)
#         sub['intersect_area'] = inter.area.fillna(0.0)
#
#         # Return only the columns declared in meta (plus geometry as active)
#         out = gpd.GeoDataFrame(
#             {
#                 'grid_id': sub['grid_id'].values,
#                 'poly_id': sub['poly_id'].values,
#                 'intersect_area': sub['intersect_area'].values,
#             },
#             geometry=sub.geometry.values,
#             crs=getattr(df, 'crs', None)
#         )
#         return out
#
#     joined_with_area = joined.map_partitions(
#         add_intersection_area,
#         right_lookup,
#         meta=meta_df
#     )
#
#     # For aggregation, select only the minimal columns to avoid metadata surprises
#     slim = joined_with_area[['grid_id', 'poly_id', 'intersect_area']]
#
#     # Global aggregation
#     aggregated = slim.groupby('grid_id').agg({
#         'poly_id': 'count',
#         'intersect_area': 'sum'
#     }).rename(columns={'poly_id': 'poly_count', 'intersect_area': 'total_intersect_area'})
#
#     grid_summary = aggregated.compute().reset_index()
#
#     # Merge back to original grid
#     grid_result = grid_gdf.merge(grid_summary, on='grid_id', how='left')
#
#     # Fill NaNs
#     grid_result[column_count] = grid_result['poly_count'].fillna(0).astype('int64')
#     grid_result[column_area] = grid_result['total_intersect_area'].fillna(0.0)
#
#     # Drop intermediate cols
#     for c in ['poly_count', 'total_intersect_area']:
#         if c in grid_result.columns:
#             grid_result = grid_result.drop(columns=c)
#
#     grid_result.to_file(save_path)
#     print(datetime.now(), f'Saved to {save_path}')

def process_grid_chunk(grid_chunk, poly_gdf, column_count, column_area):

    # Spatial join: grid cell intersects which polygons?
    joined = gpd.sjoin(grid_chunk, poly_gdf, how='left', predicate='intersects')

    # For each grid cell, count polygons and sum intersection area
    def calc_area(row):
        if pd.isna(row['index_right']):
            return 0.0
        poly_geom = poly_gdf.iloc[int(row['index_right'])].geometry
        return row.geometry.intersection(poly_geom).area

    joined[column_area] = joined.apply(calc_area, axis=1)
    summary = joined.groupby('grid_id').agg(
        **{column_count: ('index_right', lambda x: x.notna().sum())},
        **{column_area: (column_area, 'sum')}
    ).reset_index()

    # Merge back to grid_chunk to ensure all cells included
    result = grid_chunk.merge(summary, on='grid_id', how='left')
    result[column_count] = result[column_count].fillna(0).astype(int)
    result[column_area] = result[column_area].fillna(0)
    result = result.drop('grid_id', axis=1)
    return result


def calculate_poly_count_area_in_each_grid_parallel(grid_vector, in_poly_vector, save_path, column_pre_name='poly',
                                                    b_poly_bounds=False, n_workers=16, b_save_numpy=False):
    """
    Parallel calculation of polygon count and intersection area in each grid cell using multiprocessing.
    """

    column_count, column_area = check_input_vector_files(grid_vector,in_poly_vector,column_pre_name,save_path)
    if column_count is None:
        return

    t0 = time.time()

    grid_gpd = gpd.read_file(grid_vector,engine="pyogrio")
    poly_gpd = gpd.read_file(in_poly_vector,engine="pyogrio")

    t1 = time.time()
    print(f'Load two vector file, cost {t1-t0} seconds')

    if b_poly_bounds:
        poly_gpd['geometry'] = poly_gpd.bounds.apply(lambda row: box(row.minx, row.miny, row.maxx, row.maxy), axis=1)

    t2 = time.time()
    print(f'Applied bounding boxes, cost {t2-t1} seconds')


    if 'grid_id' not in grid_gpd.columns:
        grid_gpd = grid_gpd.reset_index().rename(columns={'index': 'grid_id'})

    t3 = time.time()
    print(f'Add grid_id, cost {t3-t2} seconds')

    # Split grid into chunks
    grid_chunks = np.array_split(grid_gpd, n_workers)

    t4 = time.time()
    print(f'Apply array split, cost {t4-t3} seconds')

    # Prepare processing function
    func = partial(
        process_grid_chunk,
        poly_gdf=poly_gpd,
        column_count=column_count,
        column_area=column_area
    )

    # Parallel processing
    with Pool(n_workers) as pool:
        results = pool.map(func, grid_chunks)

    t5 = time.time()
    print(f'Parallel processing, cost {t5-t4} seconds')

    # Concatenate all results
    final_gdf = pd.concat(results, ignore_index=True)
    # final_gdf = final_gdf.sort_values('grid_id').reset_index(drop=True)

    t6 = time.time()
    print(f'Concatenate, cost {t6-t5} seconds')

    # Save to file
    if b_save_numpy:
        column_count_array = np.array(final_gdf[column_count])
        column_area_array = np.array(final_gdf[column_area])
        np.save(f"{column_count}.npy",column_count_array)
        np.save(f"{column_area}.npy",column_area_array)
    else:
        final_gdf.to_file(save_path)
        print(datetime.now(), f'Saved to {save_path}')

    t7 = time.time()
    print(f'Saved to file, cost {t7-t6} seconds')

    # return final_gdf



def test_calculate_poly_count_area_in_each_grid():

    # grid_vector=os.path.expanduser('~/Downloads/tmp/h3_cells_for_merging/h3_cells_res8_panArctic_s2_rgb_2024_object_detection_s2_exp5_post_1.shp')
    grid_vector=os.path.expanduser('~/Downloads/tmp/h3_cells_panArctic_rts.gpkg')
    in_poly_vector = os.path.expanduser('~/Data/published_data/Dai_etal_2025_largeRTS_ArcticDEM/ArcticRTS_epsg3413_ClassInt.shp')

    save_path = 'polygon_count_area_per_grid.gpkg'
    # save_path = grid_vector
    # calculate_poly_count_area_in_each_grid(grid_vector, in_poly_vector, save_path,b_poly_bounds=True)
    # calculate_poly_count_area_in_each_grid_dask(grid_vector, in_poly_vector, save_path,b_poly_bounds=True)
    calculate_poly_count_area_in_each_grid_parallel(grid_vector, in_poly_vector, save_path,b_poly_bounds=True)



def main(options, args):

    grid_vector = args[0]
    # in_vectors = [item for item in args[1:]]
    # save_path = options.save_path
    save_path = grid_vector   # save the result into the orignal grid
    b_using_bounding_box = options.using_bounding_box
    process_num = options.process_num

    input_txt = options.input_txt
    in_vectors_colum_dict = {}
    if input_txt is not None:
        tmp_list = io_function.read_list_from_txt(input_txt)
        for tmp in tmp_list:
            col_and_file = [item.strip() for item in tmp.split(',')]
            # print(col_and_file)
            in_vectors_colum_dict[col_and_file[0]] = col_and_file[1]
    else:
        print('Please set "--input_txt"')
        return

    # if save_path != grid_vector:
    #     print('Please ')
    #     return
    b_save2numpy = True

    # save for backup
    io_function.save_dict_to_txt_json(input_txt+'.json',in_vectors_colum_dict)

    for idx, col_name in enumerate(in_vectors_colum_dict.keys()):
        in_poly_vector = in_vectors_colum_dict[col_name]
        basic.outputlogMessage(f'({idx+1}/{len(in_vectors_colum_dict)}) Working on {in_vectors_colum_dict[col_name]}')

        # print(grid_vector, in_poly_vector, save_path, col_name,b_using_bounding_box)

        # calculate_poly_count_area_in_each_grid(grid_vector, in_poly_vector, save_path, column_pre_name=col_name,
        #                                        b_poly_bounds=b_using_bounding_box)
        calculate_poly_count_area_in_each_grid_parallel(grid_vector, in_poly_vector, save_path, column_pre_name=col_name,
                                               b_poly_bounds=b_using_bounding_box,n_workers=process_num, b_save_numpy=b_save2numpy)



if __name__ == '__main__':

    # test_calculate_poly_count_area_in_each_grid()
    # sys.exit(0)

    usage = "usage: %prog [options] grid_vector "
    parser = OptionParser(usage=usage, version="1.0 2021-4-15")
    parser.description = 'Introduction: convert polygons in shapefiles to many geojson (each for one polygon).'

    # parser.add_option("-s", "--save_path",
    #                   action="store", dest="save_path",default="polygon_count_area_per_grid.gpkg",
    #                   help="the save path ")

    parser.add_option("-i", "--input_txt",
                      action="store", dest="input_txt",
                      help="the input txt contain column name and vector path (column_name, vector_path)")

    parser.add_option("-p", "--process_num",
                      action="store", dest="process_num",type=int, default=16,
                      help="the process number ")

    parser.add_option("-b", "--using_bounding_box",
                      action="store_true", dest="using_bounding_box",default=False,
                      help="whether use the boudning boxes of polygons, this can avoid some invalid"
                           " polygons and be consistent with YOLO output")


    (options, args) = parser.parse_args()
    # print(options.no_label_image)
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)
