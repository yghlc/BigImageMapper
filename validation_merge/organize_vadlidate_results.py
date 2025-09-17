#!/usr/bin/env python
# Filename: organize_vadlidate_results.py 
"""
introduction: (1) validate the h3 grid against with exist grouth truth
(2) copy the validate results in the h3 folder into the orignal h3 grid vector files

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 17 September, 2025
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
from shapely.errors import TopologicalError # Or TopologyException

import json


def save_true_positive_to_folder(result_dir, h3_id, overlap_ratio_list=[]):
    h3_grid_dir = os.path.join(result_dir, h3_id)
    if os.path.isdir(h3_grid_dir) is False:
        io_function.mkdir(h3_grid_dir)
    validate_json = os.path.join(h3_grid_dir, f'validated_{h3_id}.json')
    validate_dict = json.load(open(validate_json,'r')) if os.path.isfile(validate_json) else {}
    validate_dict['h3ID'] = h3_id
    # will overwrite if user already exists
    validate_dict['ground_truth'] = {
        'ValidateResult': 'TP',
        'targetCount': len(overlap_ratio_list)
    }
    with open(validate_json, 'w') as f:
        json.dump(validate_dict, f, indent=2)


def validate_against_ground_truth(grid_vector_path, ground_truth_shp_list, result_dir, min_overlap_per=0.3):
    '''
    checking if any ground truth fall within grid, if yes, then consider that grid as true positives.
    will save the validation result into a json file in "result_dir"
    :param grid_vector_path: the path the grid vector
    :param ground_truth_shp_list: the list (file path) of ground truth (boundaries or bounding boxes)
    :param result_dir: the directory containing h3 folder.
    :param min_overlap_per: the percentage of overlap (calculate against ground truth)
    :return:
    '''

    # checking projection
    grid_prj = vector_gpd.get_projection(grid_vector_path,format='epsg')
    for shp in ground_truth_shp_list:
        g_shp_prj = vector_gpd.get_projection(shp,format='epsg')
        if g_shp_prj != grid_prj:
            raise ValueError(f'The map projection ({grid_prj} vs {g_shp_prj}) between {grid_vector_path} and {shp} is different')

    # read all ground truth polygons (positive or class_int is 1)
    gt_box_polygons = []
    for gt_shp in ground_truth_shp_list:
        geometries, train_class = vector_gpd.read_polygons_attributes_list(gt_shp,
                                        'class_int',b_fix_invalid_polygon=False)

        sel_polys = [poly for poly, class_int in zip(geometries,train_class) if class_int==1]
        gt_box_polygons.extend(sel_polys)

    ## save for checking
    # save_circle_dict = {'id': [item+1 for item in range(len(geom_pos_circle_list))], "Polygon": geom_pos_circle_list}
    # save_pd = pd.DataFrame(save_circle_dict)
    # ref_prj = map_projection.get_raster_or_vector_srs_info_proj4(existing_data)
    # vector_gpd.save_polygons_to_files(save_pd, 'Polygon', ref_prj, 'geom_circles.shp')

    # build a tree
    tree = STRtree(gt_box_polygons)


    grid_polygons, h3_ids = vector_gpd.read_polygons_attributes_list(grid_vector_path,'h3_id_8',
                                                                     b_fix_invalid_polygon=False)

    for idx, (grid_poly, h3id) in enumerate(zip(grid_polygons, h3_ids)):
        if idx%100 == 0:
            print(f'Validating against ground truth progress: {idx+1}/{len(grid_polygons)}')

        inter_or_touch_list = tree.query(grid_poly)

        if len(inter_or_touch_list) > 0:
            area_ratio_list = []
            b_true_pos = False
            for inter_idx in inter_or_touch_list:
                try:
                    overlap = grid_poly.intersection(gt_box_polygons[inter_idx])
                    area_ratio = overlap.area / gt_box_polygons[inter_idx].area
                except TopologicalError as e:
                    basic.outputlogMessage(f"Caught a TopologicalError: {e}")
                    area_ratio = 0
                except Exception as e:  # Catch any other unexpected exceptions
                    basic.outputlogMessage(f"Caught an unexpected exception: {e}")
                    area_ratio = 0
                area_ratio_list.append(area_ratio)
                if area_ratio > min_overlap_per:
                    b_true_pos = True
            if b_true_pos:
                save_true_positive_to_folder(result_dir, h3id, area_ratio_list)


def test_validate_against_ground_truth():

    grid_vector_path = os.path.expanduser('~/Data/rts_ArcticDEM_mapping/combine_mapping_results/h3_cells_select_by_s2_result.gpkg')
    gt_path = os.path.expanduser('~/Data/Arctic/pan_Arctic/training_boxes_sentinel-2/train_set01_boxes_poly_s2_2024_v2.shp')
    ground_truth_shp_list = [gt_path]
    result_dir = os.path.expanduser('~/Data/rts_ArcticDEM_mapping/validation/data_multi_png')

    validate_against_ground_truth(grid_vector_path, ground_truth_shp_list, result_dir, min_overlap_per=0.3)

    pass


def main(options, args):
    grid_path = args[0]
    val_result_dir = args[1]

    # if set a save path, will save to antoher directory
    save_path = grid_path if options.save_path is None else options.save_path
    save_xlsx = options.save_xlsx
    ground_truths = options.ground_truths
    print('ground_truths:',ground_truths)









if __name__ == '__main__':
    test_validate_against_ground_truth()
    sys.exit(0)

    usage = "usage: %prog [options] grid_vector result_dir "
    parser = OptionParser(usage=usage, version="1.0 2025-09-17")
    parser.description = 'Introduction: organize validate results for h3 grids'

    parser.add_option("-s", "--save_path",
                      action="store", dest="save_path",
                      help="the file path for saving the results")

    parser.add_option("-x", "--save_xlsx",
                      action="store", dest="save_xlsx",
                      help="the file path for saving the statistic tables")


    parser.add_option("-e", "--ground_truths",
                      action="append", dest="ground_truths",  default=[],
                      help="the file path of ground truth, for multiple ground truth, set this option for multiple times ")


    (options, args) = parser.parse_args()
    # if len(sys.argv) < 2:
    #     parser.print_help()
    #     sys.exit(2)

    main(options, args)
