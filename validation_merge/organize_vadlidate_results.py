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

from scipy.cluster.hierarchy import leaves_list

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
# import parameters
import basic_src.basic as basic
from datetime import datetime
import basic_src.io_function as io_function
# import basic_src.map_projection as map_projection
import datasets.vector_gpd as vector_gpd
import shapely
from shapely.strtree import STRtree
from shapely.errors import TopologicalError # Or TopologyException

import json
import geopandas as gpd
import re
from pathlib import Path

h3ID_column_name = 'h3_id_8'

def save_true_positive_to_folder(result_dir, h3_id, ground_truth='ground_truth', overlap_ratio_list=[]):
    h3_grid_dir = os.path.join(result_dir, h3_id)
    if os.path.isdir(h3_grid_dir) is False:
        io_function.mkdir(h3_grid_dir)
    validate_json = os.path.join(h3_grid_dir, f'validated_{h3_id}.json')
    validate_dict = json.load(open(validate_json,'r')) if os.path.isfile(validate_json) else {}
    validate_dict['h3ID'] = h3_id
    # will overwrite if user already exists
    validate_dict[ground_truth] = {
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

    if grid_prj== 4326:
        raise ValueError('Currently, not supporting EPSG4326, as it need to buffer polygons in meters')

    grid_polygons, h3_ids = vector_gpd.read_polygons_attributes_list(grid_vector_path,h3ID_column_name,
                                                                     b_fix_invalid_polygon=False)

    # read ground truth polygons (positive or class_int is 1) in each file, then validate them
    # only check True Positive: the h3 grid containing ground truth polygons
    for gt_shp in ground_truth_shp_list:
        basic.outputlogMessage(f'Validating against the ground truth: {gt_shp}')
        geometries, train_class = vector_gpd.read_polygons_attributes_list(gt_shp,
                                        'class_int',b_fix_invalid_polygon=False)

        # If a geometry is a Point, buffer it by 10 meters; otherwise, keep it unchanged.
        geometries = [item.buffer(10) if item.type == 'Point' else item for item in geometries]

        gt_shp_basename = os.path.basename(gt_shp)
        gt_box_polygons = [poly for poly, class_int in zip(geometries,train_class) if class_int==1]
        basic.outputlogMessage(f'Read {len(gt_box_polygons)} ground truth polygons')


        # build a tree
        tree = STRtree(gt_box_polygons)

        for idx, (grid_poly, h3id) in enumerate(zip(grid_polygons, h3_ids)):
            if idx%100 == 0:
                print(f'Validating against ground truth progress: {idx}/{len(grid_polygons)}')

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
                    if area_ratio > min_overlap_per:
                        b_true_pos = True
                        area_ratio_list.append(area_ratio)

                # testing
                # if h3id=="880510a99bfffff":
                #     print(inter_or_touch_list)
                #     print(area_ratio_list)
                #     sys.exit(1)
                if b_true_pos:
                    save_true_positive_to_folder(result_dir, h3id, ground_truth=gt_shp_basename, overlap_ratio_list=area_ratio_list)


def test_validate_against_ground_truth():

    grid_vector_path = os.path.expanduser('~/Data/rts_ArcticDEM_mapping/combine_mapping_results/h3_cells_select_by_s2_result.gpkg')
    gt_path = os.path.expanduser('~/Data/Arctic/pan_Arctic/training_boxes_sentinel-2/train_set01_boxes_poly_s2_2024_v2.shp')
    ground_truth_shp_list = [gt_path]
    result_dir = os.path.expanduser('~/Data/rts_ArcticDEM_mapping/validation/data_multi_png')

    validate_against_ground_truth(grid_vector_path, ground_truth_shp_list, result_dir, min_overlap_per=0.3)

    pass

EMAIL_REGEX = re.compile(
    r"^(?=.{1,254}$)(?=.{1,64}@)"
    r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+"
    r"(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*"
    r"@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,}$"
)

# Common geospatial vector file extensions
VECTOR_EXTENSIONS = {
    ".shp",       # ESRI Shapefile geometry (usually with .dbf, .shx, etc.)
    ".dbf",       # attribute table (part of Shapefile set)
    ".shx",       # shapefile index
    ".prj",       # projection file
    ".cpg",       # code page for shapefile
    ".gpkg",      # GeoPackage
    ".geojson",   # GeoJSON
    ".json",      # often used for GeoJSON
    ".kml",       # Keyhole Markup Language
    ".kmz",       # Compressed KML
    ".gml",       # Geography Markup Language
    ".mif",       # MapInfo Interchange
    ".mid",       # MapInfo Interchange data
    ".tab",       # MapInfo TAB
    ".dgn",       # MicroStation Design
    ".dxf",       # AutoCAD DXF
    ".fgb",       # FlatGeobuf
    ".gdb",       # ESRI File Geodatabase directory (usually a folder)
    ".sqlite",    # Spatialite/SQLite
    ".csv"        # sometimes used with WKT/geometry columns
}
def is_email(s: str) -> bool:
    if not s or "@" not in s:
        return False
    return EMAIL_REGEX.match(s.strip()) is not None

def is_vector_filename(s: str) -> bool:
    if not s:
        return False
    p = Path(s.strip().strip('"').strip("'"))
    # Handle cases like filename.SHP or mixed case
    ext = p.suffix.lower()
    if ext in VECTOR_EXTENSIONS:
        return True
    # Handle double extensions like .geo.json
    # Check last two suffixes combined
    if len(p.suffixes) >= 2:
        comb = "".join(sfx.lower() for sfx in p.suffixes[-2:])
        if comb in {".geojson", ".topojson"}:
            return True
    # Some vector sources are directories (e.g., .gdb file geodatabase)
    # If path ends with .gdb (even if it’s a directory), above covers it.
    return False

def validate_type(source_str: str) -> str:
    """
    Classify input string as:
    - 'email' if it's an email address
    - 'vector_file' if it's a geospatial vector file name/path
    - 'other' otherwise
    """
    if not isinstance(source_str, str):
        return "other"
    s = source_str.strip()
    if not s:
        return "other"

    if is_email(s):
        return "email"

    if is_vector_filename(s):
        return "vector_file"

    return "other"



def save_validate_result_2_vector_file(in_vector_path, save_path,json_list):

    grid_gpd = gpd.read_file(in_vector_path)
    h3ID_list = grid_gpd[h3ID_column_name].to_list()
    gt_validate = [""]*len(h3ID_list)
    gt_count = [0]*len(h3ID_list)
    web_validate = [""]*len(h3ID_list)
    web_count = [0]*len(h3ID_list)

    for idx, js_file in enumerate(json_list):
        validate_dict = io_function.read_dict_from_txt_json(js_file)
        h3_id = validate_dict['h3ID']
        vec_idx = h3ID_list.index(h3_id)
        for key in validate_dict.keys():
            if key=='h3ID':
                continue
            if validate_type(key) == 'email':
                if len(web_validate[vec_idx]) > 0:
                    web_validate[vec_idx] += ','
                web_validate[vec_idx] += validate_dict[key]['ValidateResult']
                web_count[vec_idx] += validate_dict[key]['targetCount']
            elif validate_type(key) == 'vector_file':
                if len(gt_validate[vec_idx]) > 0:
                    gt_validate[vec_idx] += ','
                gt_validate[vec_idx] += validate_dict[key]['ValidateResult']
                gt_count[vec_idx] += validate_dict[key]['targetCount']
            else:
                raise ValueError(f'Unknown validate type in {js_file} ')

    # add to the vector file
    save_validate_dict= {'GT_Valid':gt_validate,'GT_Count':gt_count, 'Web_Valid':web_validate,'Web_Count':web_count}

    for key in save_validate_dict.keys():
        if os.path.isfile(save_path) and vector_gpd.is_field_name_in_shp(save_path,key):
            print(f'Warning, column {key} already in {save_path}, will replace it')
        grid_gpd[key] = save_validate_dict[key]
    grid_gpd.to_file(save_path)
    print(f'saved to {save_path}')


def main(options, args):
    grid_path = args[0]
    val_result_dir = args[1]

    # if set a save path, will save as another file
    save_path = grid_path if options.save_path is None else options.save_path
    save_xlsx = options.save_xlsx
    ground_truths = options.ground_truths
    print('ground_truths files:',ground_truths)
    if len(ground_truths) > 0:
        validate_against_ground_truth(grid_path, ground_truths, val_result_dir, min_overlap_per=0.3)

    # copy files in validate json to the shape folders
    validate_json_list = io_function.get_file_list_by_pattern(val_result_dir,'*/validated*.json')
    if len(validate_json_list) > 0:
        save_validate_result_2_vector_file(grid_path,save_path,validate_json_list)
    else:
        print(f'No validation json files in the sub-folders of {val_result_dir}')





if __name__ == '__main__':
    # test_validate_against_ground_truth()
    # sys.exit(0)

    usage = "usage: %prog [options] grid_vector result_dir "
    parser = OptionParser(usage=usage, version="1.0 2025-09-17")
    parser.description = 'Introduction: organize validate results for h3 grids'

    parser.add_option("-s", "--save_path",
                      action="store", dest="save_path",
                      help="the file path for saving the results")

    parser.add_option("-x", "--save_xlsx",
                      action="store", dest="save_xlsx",
                      help="the file path for saving the statistic tables")


    parser.add_option("-g", "--ground_truths",
                      action="append", dest="ground_truths",  default=[],
                      help="the file path of ground truth, for multiple ground truth, set this option for multiple times ")


    (options, args) = parser.parse_args()
    # if len(sys.argv) < 2:
    #     parser.print_help()
    #     sys.exit(2)

    main(options, args)
