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

import basic_src.io_function as io_function

def short_h3(h3id):
    return h3id.rstrip('f')

def unshort_h3(short_id):
    return short_id.ljust(15, 'f')

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


def main():
    # test_get_h3_cell_id()
    # test_test_get_h3_cell_id()
    # test_get_grid_distance_between_cells()
    test_select_isolated_files_h3_at_res()
    pass


if __name__ == '__main__':
    main()
