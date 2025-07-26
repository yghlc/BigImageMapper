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




def main():
    # test_get_h3_cell_id()
    test_test_get_h3_cell_id()
    pass


if __name__ == '__main__':
    main()
