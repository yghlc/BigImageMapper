#!/usr/bin/env python
# Filename: create_grid_polygons_perma 
"""
introduction: based on permafrost map, to create a many polygons for downloading images and creating mosaic

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 08 August, 2020
"""

import os,sys

deeplabforRS =  os.path.expanduser('~/codes/PycharmProjects/DeeplabforRS')
sys.path.insert(0, deeplabforRS)
import vector_gpd
import basic_src.map_projection as map_projection
import basic_src.io_function as io_function

import pandas as pd

shp_dir = os.path.expanduser('~/Data/Qinghai-Tibet/all_permafrost_areas_QTP/extent')

def main():

    # grid polygons
    qtp_grid_50km = os.path.join(shp_dir,'sub_images_grid.shp')
    # main permaforst areas based on permafrost map, has been pre-processed: remove small ones, simply the boundaries
    qtp_main_perma_area_simp = os.path.join(shp_dir,'zou_Albers_main_permafrost_edit_simp1km.shp')

    # qtp_grid_50km and qtb_main_perma_area_simp have the same projection
    grid_prj = map_projection.get_raster_or_vector_srs_info_proj4(qtp_grid_50km)
    perma_area_prj = map_projection.get_raster_or_vector_srs_info_proj4(qtp_main_perma_area_simp)
    if grid_prj != perma_area_prj:
        raise ValueError('%s and %s do not have the same projection'%(grid_prj,perma_area_prj))

    grids  = vector_gpd.read_polygons_gpd(qtp_grid_50km)
    perma_areas = vector_gpd.read_polygons_gpd(qtp_main_perma_area_simp)

    perma_size_list = vector_gpd.read_attribute_values_list(qtp_main_perma_area_simp,'Area_km2')

    small_perma_areas_list = []

    for idx, (perma_poly,size) in enumerate(zip(perma_areas,perma_size_list)):
        print(' processing %dth permafrost area'%idx)
        # if the permafrost area is < 50*50 km^2, then do not split it to smaller ones.
        if size < 2500:
            small_perma_areas_list.append(perma_poly)
            continue

        # split the big permafrost area into many small ones
        for grid in grids:
            inte_res = perma_poly.intersection(grid)
            if inte_res.is_empty is False:

                inte_res_multi = vector_gpd.MultiPolygon_to_polygons(idx,inte_res)

                for tmp in inte_res_multi:
                    # remove holes if they exist
                    small_ones = vector_gpd.fill_holes_in_a_polygon(tmp)
                    small_perma_areas_list.append(small_ones)


    # save
    save_path = io_function.get_name_by_adding_tail(qtp_main_perma_area_simp,'small')
    save_polyons_attributes = {}
    save_polyons_attributes["Polygons"] = small_perma_areas_list

    # wkt_string = map_projection.get_raster_or_vector_srs_info_wkt(qtp_main_perma_area_simp)
    wkt_string = map_projection.get_raster_or_vector_srs_info_proj4(qtp_main_perma_area_simp)
    polygon_df = pd.DataFrame(save_polyons_attributes)
    vector_gpd.save_polygons_to_files(polygon_df, 'Polygons', wkt_string, save_path)




    pass

if __name__ == "__main__":
    main()