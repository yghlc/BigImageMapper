#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Introduction: the prediction scripts (e.g., sam_dir/sam_predict.py) in this project were designed for large images, 
and they may not work well for small images (quite slow, as need to reload the train models for each image). 
This script is used to mosaic small images into a large image, and then the prediction scripts can be used for the large image.

1. for small images from H3 grids (8th resolution), we can group them into a larger H3 grid (e.g., 4th resolution), 
and then mosaic the small images in each group into a large image.  

2. for other small images, we can using a reference grid systems (e.g, 20 km by 20 km grid or 10 km by 10 km grid) to group the small images, 
and then mosaic the small images in each group into a large image.

Author: Huang Lingcao
Email: huanglingcao@gmail.com
Created: 2026-09-28
"""

import os, sys
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters

import basic_src.io_function as io_function
import basic_src.map_projection as map_projection
import basic_src.basic as basic
import datasets.raster_io as raster_io
import datasets.vector_gpd as vector_gpd
import datasets.geo_index_h3 as geo_index_h3


def mosaics_images_vrt(raster_files,outputfile,nodata=None, resampling_method='average'):
    """
    mosaic a set of images using gdalbuildvrt. All the images must be in the same coordinate system and have a matching number of bands,
    Args:
        raster_files:a set of images with same coordinate system and have a matching number of bands, list type
        outputfile: the mosaic result file

    Returns: the result path if successful, False otherwise

    """
    if isinstance(raster_files,list) is False:
        raise ValueError('the type of raster_files must be list')
    if len(raster_files)<2:
        raise ValueError(f'file count less than 2: {raster_files}')

    cmd_str = f"gdalbuildvrt -resolution average -r {resampling_method} "
    # add nodata
    if nodata is not None:
        cmd_str += f' -srcnodata {nodata} '
        cmd_str += f' -vrtnodata {nodata} '

    path_no_ext = os.path.splitext(outputfile)[0]
    img_list_txt = f'{path_no_ext}_infile_list.txt'
    io_function.save_list_to_txt(img_list_txt, raster_files)

    cmd_str += f" -input_file_list {img_list_txt} {outputfile}"

    return basic.exec_command_string_one_file(cmd_str,outputfile)


def group_mosaic_by_ref_grid():
    # to implement
    pass



def group_mosaic_by_h3_grid(small_img_dir, save_dir, low_h3_res=8, high_h3_res=4):
    # to group small images and create mosaic for them based on h3 grid

    if os.path.isdir(save_dir) is False:
        io_function.mkdir(save_dir)

    # low_h3_res: should match these H3 id in the filename
    small_img_list = io_function.get_file_list_by_pattern(small_img_dir,'*.tif')

    if len(small_img_list) == 0:
        print(f"No TIFF files found in directory: {small_img_dir}")
        return 

    # get the H3 id of each file name 
    small_img_h3_id_list = [geo_index_h3.get_h3_id_from_filename(img_path) for img_path in small_img_list]

    # group the file name by high_h3_res
    h3_id_file_group = {}
    for idx_h3_id, img_path in zip(small_img_h3_id_list, small_img_list):
        parent_id = geo_index_h3.get_h3_parent(idx_h3_id,high_h3_res)
        # change to absolution path, otherwise, may end in error in later steps
        h3_id_file_group.setdefault(parent_id, []).append(os.path.abspath(img_path)) 

    # save h3_id_file_group to txt
    h3_id_file_group_txt = os.path.join(save_dir,f'h3_{high_h3_res}_id_file_group.txt')
    io_function.save_dict_to_txt_json(h3_id_file_group_txt,h3_id_file_group)
    
    # create mosaic for each group by building VRT iin the save dir
    for parent_id, img_paths in h3_id_file_group.items():
        vrt_path = os.path.join(save_dir, f"h3_{high_h3_res}_id_{parent_id}_vrt.tif")
        basic.outputlogMessage(f"Creating VRT mosaic for H3 parent {parent_id} with {len(img_paths)} input files.")

        if len(img_paths) < 2:
            # create a soft link
            os.symlink(img_paths[0], vrt_path)
        else:
            no_data = raster_io.get_nodata(img_paths[0])
            mosaics_images_vrt(img_paths,vrt_path,nodata=no_data, resampling_method='average')




def main(options, args):
    small_img_dir = args[0]
    save_dir = args[1]
    b_h3_grid = options.b_h3_grid # to h3 grid in filename, and will use it
    reference_grids_shp = options.reference_grids_shp # if not None, to use reference grid

    if b_h3_grid:  
        high_h3_resolution = options.ref_h3_resolution
        group_mosaic_by_h3_grid(small_img_dir, save_dir, low_h3_res=8, high_h3_res=high_h3_resolution)

     


if __name__ == "__main__":
    usage = "usage: %prog [options]  input_dir output_dir "
    parser = OptionParser(usage=usage, version="1.0 2026-09-28")
    parser.description = 'Introduction: group many small images and create mosaic images for each group'

    parser.add_option("-r", "--reference_grids_shp",
                      action="store", dest="reference_grids_shp",
                      help="the vector file of reference grids")

    parser.add_option("-g", "--b_h3_grid",
                    action="store_true", dest="b_h3_grid", default=False,
                    help="if set, it means H3 IDs are in filenames, and the H3 grid system will be used")
    
    parser.add_option("-l", "--ref_h3_resolution",
                    action="store", dest="ref_h3_resolution", type=int, default=4,
                    help="the reference (higher) resolution of H3 grids")


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) != 2:
        parser.print_help()
        sys.exit(2)
    main(options, args)