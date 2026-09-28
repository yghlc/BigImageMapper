#!/usr/bin/env python
# Filename: geo_index_h3.py 
"""
introduction:  using Uber's H3 Hexagonal Hierarchical Geospatial Indexing System

github (python): https://github.com/uber/h3?tab=readme-ov-file
site: https://h3geo.org/docs/

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 24 July, 2025
"""

import os
import h3
import geopandas as gpd
from collections import Counter

import shapely
from shapely.geometry import Polygon, MultiPolygon, LineString, GeometryCollection
from shapely.ops import split

import antimeridian

import re
from pathlib import Path

import basic_src.io_function as io_function

def short_h3(h3id):
    return h3id.rstrip('f')

def unshort_h3(short_id):
    return short_id.ljust(15, 'f')

# Convert Shapely polygon to H3 LatLngPoly
def xy_to_latlng(coords):
    return [(y, x) for x, y in coords]  # shapely: (x=lng, y=lat) -> H3: (lat, lng)

def latlng_to_xy(coords):
    return [(y, x) for x, y in coords]  # H3: (lat, lng) ->  shapely: (x=lng, y=lat)

def get_folder_file_save_path(root, latitude, longitude, res=14, extension='.tif'):
    """
    Generate a unique file path for saving a file, using H3 cell IDs as folder and file names.

    The path structure is:
        root/<short_h3_id_res1>/<short_h3_id_res2>/.../<short_h3_id_lastres>.EXT

    If the file already exists, a numeric suffix (_1, _2, ...) is appended to the file name.

    Parameters
    ----------
    root : str
        The root directory for saving the file.
    latitude : float
        Latitude of the location.
    longitude : float
        Longitude of the location.
    res : int or list of int, optional
        H3 resolution(s) to use for generating the folder structure and file name.
        If a list, each resolution creates a subfolder, and the last is used as the file name.
        Default is 14.
    extension : str, optional
        File extension (with or without leading dot). Default is '.tif'.

    Returns
    -------
    str
        A full file path (with folders) that does not exist yet.

    Notes
    -----
    - The function does **not** create directories; it only generates the path.
    - If the file already exists, a numeric suffix is appended to the file name.
    """
    # Ensure extension starts with '.'
    if not extension.startswith('.'):
        extension = '.' + extension

    # Normalize resolutions to list
    if not isinstance(res, (list, tuple)):
        res_list = [res]
    else:
        res_list = list(res)

    # Get short H3 IDs for each resolution
    h3_ids = [short_h3(get_h3_cell_id(latitude, longitude, r)) for r in res_list]

    # Build the folder path and file name
    h3_ids.insert(0, root)
    folder_file_name = os.path.join(*h3_ids)
    file_path = folder_file_name + extension

    # Ensure unique file name if file already exists
    same_file_name_n = 0
    while os.path.isfile(file_path):
        same_file_name_n += 1
        file_path = f"{folder_file_name}_{same_file_name_n}{extension}"

    return file_path


def get_h3_parent(in_h3_id, out_res):
    if not h3.is_valid_cell(in_h3_id):
        raise ValueError(f"H3 cell is invalid: {in_h3_id}")

    curr_res = h3.get_resolution(in_h3_id)

    if curr_res <= out_res:
        raise ValueError(f"Current resolution ({curr_res}) must be greater than target resolution ({out_res}).")

    return h3.cell_to_parent(in_h3_id, out_res)

def get_h3_id_from_filename(file_path):
    name = Path(file_path).name
    # most regular h3 id is start with 8, edge or invalid are start with 4 or b (need to ignore or handle)
    matches = re.findall(r'(8[0-9a-f]{14})', name, re.IGNORECASE)

    if len(matches) == 0:
        raise ValueError(f"No H3 ID found in filename: {file_path}")
    if len(matches) > 1:
        raise ValueError(f"Multiple H3 IDs found in filename: {file_path}")

    return matches[0]


def get_h3_cell_id(latitude, longitude, resolution):
    """
    Given a latitude, longitude, and H3 resolution, return the H3 cell ID as a string.
    Args:
        latitude (float): Latitude in decimal degrees.
        longitude (float): Longitude in decimal degrees.
        resolution (int): H3 resolution (0-15), refer to: https://h3geo.org/docs/core-library/restable
    Returns:
        str: H3 cell ID.
    """
    return h3.latlng_to_cell(latitude, longitude, resolution)

def test_get_h3_cell_id():
    # for res in range(0,16):
    #     cell_id = get_h3_cell_id(37.775938728915946, -122.41795063018799, res)
    #     print(res,':',cell_id)  # e.g., '87283082bffffff'

    lat1, lng1 = 75.0, -42.0
    lat2, lng2 = 75.0005, -42.0005  # Very close by
    cell1 = h3.latlng_to_cell(lat1, lng1, 7)
    cell2 = h3.latlng_to_cell(lat2, lng2, 7)
    print(cell1)
    print(cell2)
    print(cell1 == cell2)  # True if they're in the same H3 cell

def test_test_get_h3_cell_id():
    lat1, lng1 = 75.0, -42.0
    root = os.path.expanduser('~/Data')
    # get_folder_file_save_path(root, lat1, lng1, res=14, extension='.tif')
    file_path = get_folder_file_save_path(root, lat1, lng1, res=[2,6,10,14], extension='.tif')
    print(file_path)



def select_isolated_files_h3(h3_id_list, threshold_grid_k):
    """
    Finds isolated H3 IDs: for each, if the total count of itself and all its
    neighbors (within threshold_grid_k) is 1, it's considered isolated.
    """
    cell_count = Counter(h3_id_list)
    io_function.save_dict_to_txt_json('cell_count.txt',cell_count)
    isolated = []
    isolated_idx = []


    for idx, h3_id in enumerate(h3_id_list):
        neighborhood = h3.grid_disk(h3_id, threshold_grid_k)
        # cell_count is a Counter object, which is a subclass of dict.
        # If you ask for a key (an H3 ID) that does not exist in the Counter,
        # it returns 0 instead of raising a KeyError.
        total_count = sum(cell_count[neighbor] for neighbor in neighborhood)
        if total_count == 1:
            isolated.append(h3_id)
            isolated_idx.append(idx)
    return isolated, isolated_idx

def select_isolated_files_h3_at_res(h3_id_list, threshold_grid_k, res):
    """
    Converts all H3 IDs to the specified resolution (keeping duplicates),
    then finds isolated IDs at that resolution.

    Args:
        h3_id_list (list): List of H3 indexes (strings).
        threshold_grid_k (int): Neighborhood k for isolation.
        res (int): Target H3 resolution.

    Returns:
        list: Isolated H3 IDs at the specified resolution (duplicates preserved).
    """
    converted_ids = []
    for h in h3_id_list:
        # print(h)
        if not h3.is_valid_cell(h):
            raise ValueError(f'cell: {h} is invalid')
        curr_res = h3.get_resolution(h)

        if curr_res > res:
            converted_ids.append(h3.cell_to_parent(h, res))
        elif curr_res == res:
            converted_ids.append(h)
        # If curr_res < res, ignore (or handle as needed for your use case)
        else:
            raise ValueError(f'current res: {curr_res} is smaller than targeted res: {res}')

        # print(curr_res, res)
    # print(converted_ids[0],converted_ids[1])
    # print('debugging, h3_id_list, threshold_grid_k, res', len(h3_id_list), threshold_grid_k, res)
    # print('debugging, converted_ids', len(converted_ids))
    c_isolate_ids, c_isolated_idx = select_isolated_files_h3(converted_ids, threshold_grid_k)

    # get the IDs  at the original res
    sel_ids = [h3_id_list[idx] for idx in c_isolated_idx]

    return sel_ids, c_isolated_idx


def test_select_isolated_files_h3_at_res():
    data_dir = os.path.expanduser('~/Data/rts_ArcticDEM_mapping/extract_sub_images')
    # txt = os.path.join(data_dir,'sub_images_labels_list.txt')
    txt = os.path.join(data_dir,'sub_images_labels_list_10000.txt')
    h3_ids_list = [os.path.splitext(os.path.basename(item))[0] for item in  io_function.read_list_from_txt(txt)]
    h3_ids_list = [item.split('_')[0]  for item in h3_ids_list] # for case like: 8e05100e9b8e317_1, remove "_1"
    # in default, we use shorted function
    h3_ids_list = [unshort_h3(item) for item in h3_ids_list]
    print(h3_ids_list[0], h3_ids_list[1], h3_ids_list[2])

    threshold_grid_k = 1
    res = 8
    # res = 14
    isolated_ids, _ = select_isolated_files_h3_at_res(h3_ids_list, threshold_grid_k, res)
    io_function.save_list_to_txt('isolated_ids.txt',isolated_ids)


def get_grid_distance_between_cells(cell1, cell2):
    if h3.get_resolution(cell1) != h3.get_resolution(cell2):
        raise ValueError("Cells are not at the same resolution!")

    if not h3.is_valid_cell(cell1) or not h3.is_valid_cell(cell2):
        raise ValueError("Invalid H3 index!")

    # Calculate the grid distance
    distance = h3.grid_distance(cell1, cell2)
    print(f"The H3 grid distance between the two cells is: {distance}")

def test_get_grid_distance_between_cells():
    # Define two H3 cell IDs at the same resolution
    lat1, lon1 = 40.7128, -74.0060
    lat2, lon2 = 34.0522, -74.2437
    # h3_cell1 = h3.geo_to_h3(, 9)  # New York City
    # h3_cell2 = h3.geo_to_h3(, 9)  # Los Angeles

    cell1 = get_h3_cell_id(lat1,lon1,9)
    cell2 = get_h3_cell_id(lat2,lon2,9)

    get_grid_distance_between_cells(cell1, cell2)


def to_lonlat_list(coords_latlon):
    # h3 returns [(lat, lon), ...]; convert to [[lon, lat], ...]
    return [[float(lon), float(lat)] for lat, lon in coords_latlon]

def close_ring(coords):
    if not coords:
        return coords
    if coords[0][0] != coords[-1][0] or coords[0][1] != coords[-1][1]:
        return coords + [coords[0][:]]
    return coords

def unwrap_ring(coords):
    # Keep consecutive longitudes within 180 degrees
    if not coords:
        return coords
    out = [coords[0][:]]
    for lon, lat in coords[1:]:
        prev_lon = out[-1][0]
        lon_adj = lon
        while lon_adj - prev_lon > 180:
            lon_adj -= 360
        while lon_adj - prev_lon < -180:
            lon_adj += 360
        out.append([lon_adj, lat])
    return out

def wrap_lon_180(lon):
    return ((lon + 180) % 360) - 180

def wrap_coords_180(coords):
    return [[wrap_lon_180(lon), lat] for lon, lat in coords]

def split_at_dateline(geom):
    minx, _, maxx, _ = geom.bounds
    if maxx <= 180 or minx >= 180:
        return [geom]
    dateline = LineString([(180.0, -90.0), (180.0, 90.0)])
    parts = split(geom, dateline)
    if isinstance(parts, GeometryCollection):
        return list(parts.geoms)
    return [parts]

def h3_cell_to_valid_geom(cell_id):
    boundary_latlon = h3.cell_to_boundary(cell_id)  # [(lat, lon), ...]
    if not boundary_latlon or len(boundary_latlon) < 3:
        return None

    ring = to_lonlat_list(boundary_latlon)  # [[lon, lat], ...]
    ring = close_ring(ring)
    ring_unwrapped = unwrap_ring(ring)

    poly = Polygon(ring_unwrapped)
    if poly.is_empty:
        return None

    minx, _, maxx, _ = poly.bounds
    if minx < 180 < maxx:
        parts = split_at_dateline(poly)
        wrapped_parts = []
        for p in parts:
            if p.is_empty:
                continue
            coords_wrapped = wrap_coords_180(list(p.exterior.coords))
            p_wrapped = Polygon(close_ring(coords_wrapped)).buffer(0)
            if not p_wrapped.is_empty:
                wrapped_parts.append(p_wrapped)
        if not wrapped_parts:
            return None
        if len(wrapped_parts) == 1:
            return wrapped_parts[0]
        return MultiPolygon(wrapped_parts)
    else:
        coords_wrapped = wrap_coords_180(list(poly.exterior.coords))
        return Polygon(close_ring(coords_wrapped)).buffer(0)

# def get_polygon_of_h3_cell(h3_id_list, map_prj='EPSG:4326', h3_id_col_name='h3_id'):
#     """
#     Get the polygon geometry of H3 cells as a GeoDataFrame.
#     Handles antimeridian-crossing cells robustly.
#
#     Args:
#         h3_id_list: Iterable of H3 cell IDs.
#         map_prj: Desired CRS for output GeoDataFrame (default EPSG:4326).
#     """
#     geoms = []
#     save_h3_ids = []
#     for cid in h3_id_list:
#         try:
#             geom = h3_cell_to_valid_geom(cid)
#         except Exception as e:
#             print(f'Warning: Failed to build polygon for id {cid}: {e}')
#             geom = None
#
#         if geom and geom.is_valid and not geom.is_empty:
#             geoms.append(geom)
#             save_h3_ids.append(cid)
#         else:
#             if geom:
#                 fixed = geom.buffer(0)
#                 if fixed.is_valid and not fixed.is_empty:
#                     geoms.append(fixed)
#                     save_h3_ids.append(cid)
#                     continue
#             print(f'Warning: The cell polygon is empty or invalid, with id: {cid}')
#
#
#     gdf = gpd.GeoDataFrame(geometry=geoms, data={h3_id_col_name:save_h3_ids}, crs='EPSG:4326')
#     if map_prj and map_prj != 'EPSG:4326' and not gdf.empty:
#         gdf = gdf.to_crs(map_prj)
#     return gdf


def get_polygon_of_h3_cell(h3_id_list, map_prj='EPSG:4326', h3_id_col_name='h3_id'):
    """
    Get the polygon geometry of an H3 cell as a GeoDataFrame.

    Args:
        h3_id (str): The list of H3 cell ID.
        map_prj (str): The desired coordinate reference system (CRS) for the output GeoDataFrame.
    """
    poly_list = []
    save_h3_ids = []
    for cid in h3_id_list:
        boundary = h3.cell_to_boundary(cid)  # [(lat, lng), ...]
        if not boundary or len(boundary) < 3:
            continue

        boundary = latlng_to_xy(boundary)

        # Ensure closed ring for Shapely
        if boundary[0] != boundary[-1]:
            boundary = boundary + [boundary[0]]

        poly = Polygon(boundary)

        if poly.is_empty:
            print(f'Warning: The cell polygon is empty: ({poly}), with id: {cid}')
            continue

        if not poly.is_valid:
            print(f'Warning: The cell polygon is invalid: ({poly}), with id: {cid}, try to fix it using antimeridian')
            poly = antimeridian.fix_polygon(poly)

        if poly.is_valid:
            poly_list.append(poly)
            save_h3_ids.append(cid)
        else:
            print(f'Warning: The cell polygon is empty or invalid ({poly}), with id: {cid}')

    # conver to GeoDataFrame
    poly_list = gpd.GeoDataFrame(geometry=poly_list, data={h3_id_col_name:save_h3_ids}, crs='EPSG:4326')
    if map_prj != 'EPSG:4326':
        poly_list = poly_list.to_crs(map_prj)

    return poly_list



def test_get_polygon_of_h3_cell():
    h3_id = '830d86fffffffff'
    h3_grid_gpd = get_polygon_of_h3_cell([h3_id], map_prj='EPSG:4326', h3_id_col_name='h3_id')
    h3_grid_gpd.to_file('test_get_polygon_of_h3_cell.gpkg')


def main():
    # test_get_h3_cell_id()
    # test_test_get_h3_cell_id()
    # test_get_grid_distance_between_cells()
    # test_select_isolated_files_h3_at_res()
    test_get_polygon_of_h3_cell()
    pass


if __name__ == '__main__':
    main()
