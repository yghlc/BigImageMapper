#!/usr/bin/env python
# Filename: create_grid_polygons_perma 
"""
introduction: based on a map, to create a many polygons for downloading images and creating mosaic

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 08 November, 2020
"""

import os,sys

deeplabforRS =  os.path.expanduser('~/codes/PycharmProjects/DeeplabforRS')
sys.path.insert(0, deeplabforRS)
import vector_gpd
import basic_src.map_projection as map_projection
import basic_src.io_function as io_function

import pandas as pd

shp_dir = os.path.expanduser('~/Data/LandCover_LandUse_Change/Potential_areas_for_preliminary_study/areas_extent')

def main():

    # grid polygons
    grid_50km = os.path.join(shp_dir,'PAMPA_outline_utm_50grid.shp')

    # main permaforst areas based on permafrost map, has been pre-processed: remove small ones, simply the boundaries
    main_area_simp = os.path.join(shp_dir,'PAMPA_outline_utm.shp')

    # qtp_grid_50km and qtb_main_perma_area_simp have the same projection
    grid_prj = map_projection.get_raster_or_vector_srs_info_proj4(grid_50km)
    perma_area_prj = map_projection.get_raster_or_vector_srs_info_proj4(main_area_simp)
    if grid_prj != perma_area_prj:
        raise ValueError('%s and %s do not have the same projection'%(grid_prj,perma_area_prj))

    grids  = vector_gpd.read_polygons_gpd(grid_50km)
    perma_areas = vector_gpd.read_polygons_gpd(main_area_simp)

    # perma_size_list = vector_gpd.read_attribute_values_list(qtp_main_perma_area_simp,'Area_km2')

    small_perma_areas_list = []

    for idx, perma_poly in enumerate(perma_areas):
        print(' processing %dth permafrost area'%idx)
        # if the permafrost area is < 50*50 km^2, then do not split it to smaller ones.
        # if size < 2500:
        #     perma_poly = vector_gpd.fill_holes_in_a_polygon(perma_poly)
        #     small_perma_areas_list.append(perma_poly)
        #     continue

        # split the big permafrost area into many small ones
        for grid in grids:
            inte_res = perma_poly.intersection(grid)
            if inte_res.is_empty is False:

                inte_res_multi = vector_gpd.MultiPolygon_to_polygons(idx,inte_res)

                for tmp in inte_res_multi:
                    # remove holes if they exist
                    small_ones = vector_gpd.fill_holes_in_a_polygon(tmp)
                    #################################
                    # we should remove some really small polygons (< 1 km^2)
                    small_perma_areas_list.append(small_ones)

    ##############################
    # have to manually merge small polygons in QGIS to its adjacent ones.


    # save
    save_path = io_function.get_name_by_adding_tail(main_area_simp,'small')
    save_path = os.path.join(shp_dir,os.path.basename(save_path))

    save_polyons_attributes = {}
    save_polyons_attributes["Polygons"] = small_perma_areas_list

    # wkt_string = map_projection.get_raster_or_vector_srs_info_wkt(qtp_main_perma_area_simp)
    wkt_string = map_projection.get_raster_or_vector_srs_info_proj4(main_area_simp)
    polygon_df = pd.DataFrame(save_polyons_attributes)
    vector_gpd.save_polygons_to_files(polygon_df, 'Polygons', wkt_string, save_path)



    pass

if __name__ == "__main__":
    main()