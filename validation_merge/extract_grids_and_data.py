#!/usr/bin/env python
# Filename: extract_grids_and_data.py 
"""
introduction: extract images, vector for each grid (cell) for validation

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 04 September, 2025
"""

import os,sys
import time
from optparse import OptionParser



code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)

import basic_src.io_function as io_function
import basic_src.basic as basic
import datasets.vector_gpd as vector_gpd
import datasets.raster_io as raster_io
from shapely.geometry import Polygon, MultiPolygon

import parameters

import numpy as np
import geopandas as gpd
from datetime import datetime
import re
import json

import bim_utils
from utility.rename_subImages import rename_sub_images

from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool

def get_mapping_shp_raster_dict(pre_names, mapping_res_ini):
    mapping_shp_raster_dict = {}
    for p_name in pre_names:
        mapping_shp_raster_dict[p_name] = {}
        for v in ['-shp', '-dir', '-pattern']:
            ini_var_name = p_name + v
            if ini_var_name.endswith('-shp'):
                # shp_path = parameters.get_file_path_parameters(mapping_res_ini,ini_var_name)
                shp_path = parameters.get_file_path_parameters_None_if_absence(mapping_res_ini,ini_var_name)
                mapping_shp_raster_dict[p_name][ini_var_name] = shp_path
            if ini_var_name.endswith('-dir'):
                dir_path = parameters.get_directory_None_if_absence(mapping_res_ini, ini_var_name)
                mapping_shp_raster_dict[p_name][ini_var_name] = dir_path
            if ini_var_name.endswith('pattern'):
                pattern = parameters.get_string_parameters_None_if_absence(mapping_res_ini, ini_var_name)
                mapping_shp_raster_dict[p_name][ini_var_name] = pattern

    return mapping_shp_raster_dict

def get_h3_id_in_sub_images_f(file_path):
    ids = re.findall(r'id([0-9a-fA-F]+)_', os.path.basename(file_path))
    return ids[0]

def get_year_in_file_or_str(file_path):
    years = re.findall(r"\d{4}", os.path.basename(file_path))
    return years[0]

def obtain_multi_data(grid_gpd, grid_vector_path,mapping_shp_raster_dict,out_dir, buffersize=10, process_num=4,
                      b_get_subImg_only=False):

    dstnodata = 0
    rectangle_ext = True
    b_keep_org_file_name = True
    sub_image_format = 'GTIFF'
    img_file_extension = '.tif'
    # sub_image_format = 'PNG'  # png dont support int16 (elevation difference)
    # img_file_extension = '.png'

    # get sub-images for each raster, and the corresponding vectors
    for set_name in mapping_shp_raster_dict.keys():
        # extract sub-images
        # train_grids_shp,image_dir, buffersize,image_or_pattern,extract_img_dir,dstnodata,process_num,rectangle_ext,b_keep_org_file_name
        img_dir = mapping_shp_raster_dict[set_name][set_name+'-dir']
        img_pattern = mapping_shp_raster_dict[set_name][set_name+'-pattern']
        if img_dir is None or img_pattern is None:
            print(f'Warning: the image dir or pattern for {set_name} is None, skip it')
            continue
        sub_image_dir = os.path.join(out_dir, set_name)
        done_indicator = os.path.join(sub_image_dir,f'{set_name}.done')
        if os.path.isfile(done_indicator):
            print(f'Sub-images for {set_name} has been extracted, skip')
            continue
        #extract sub-images
        bim_utils.extract_sub_images(grid_vector_path,img_dir,buffersize,img_pattern,sub_image_dir,dstnodata,process_num,rectangle_ext,b_keep_org_file_name,
                                     save_format=sub_image_format)

        # rename the sub-images
        sub_images_dir2 = os.path.join(sub_image_dir,'subImages')
        rename_sub_images(grid_vector_path, sub_images_dir2, img_file_extension, set_name, 'h3_id_8')
        print(datetime.now(), f'Extracted sub-images for {set_name}, saved in {sub_image_dir}')

        io_function.save_text_to_file(done_indicator, f'Complete extracting sub-images at {datetime.now()}')

    if b_get_subImg_only:
        print(print(datetime.now(), f'Only getting sub-images {b_get_subImg_only}, skip getting polygons and organize into h3 folders'))
        return

    shp_gpd_dict = {}
    for set_name in mapping_shp_raster_dict.keys():
        shp_path = mapping_shp_raster_dict[set_name][set_name + '-shp']
        if shp_path is None:
            continue
        io_function.is_file_exist(shp_path)
        shp_gpd_dict[set_name] = gpd.read_file(shp_path)

    # save the grids in grid_gpd to geojson format
    vector_save_dir = out_dir
    save_h3_id_list = []
    for idx, row in grid_gpd.iterrows():
        h3_id = row["h3_id_8"]
        cell_vec_dir = os.path.join(vector_save_dir, h3_id)
        if os.path.isdir(cell_vec_dir):
            print(f'Waning, folder: {h3_id} exists, skip getting vector files')
            continue
        io_function.mkdir(cell_vec_dir)
        single_gdf = grid_gpd.iloc[[idx]]

        # save the cell grids
        cell_save_p = os.path.join(cell_vec_dir, f"h3_cell_{h3_id}.geojson")
        single_gdf.to_file(cell_save_p, driver="GeoJSON")
        save_h3_id_list.append(h3_id)

        # for shp dataset
        for set_name in shp_gpd_dict.keys():
            count = row[set_name+"_C"]
            if count < 1:
                continue
            shp_poly_save_p = os.path.join(cell_vec_dir, f"{set_name}_{h3_id}.geojson")
            # extract the corresponding vectors
            overlap_touch = gpd.sjoin(shp_gpd_dict[set_name], single_gdf, how='inner', predicate='intersects')
            # remove duplicated geometries in overlap_touch
            overlap_touch = overlap_touch.drop_duplicates(subset=['geometry'])  # only check geometry
            overlap_touch.to_file(shp_poly_save_p,driver="GeoJSON")

    # print(mapping_shp_raster_dict)

    # organize the sub-images
    sub_img_id_path_dict = {}
    for set_name in mapping_shp_raster_dict.keys():
        img_dir = mapping_shp_raster_dict[set_name][set_name + '-dir']
        if img_dir is None:
            continue
        sub_image_dir = os.path.join(out_dir, set_name,'subImages')
        sub_image_list = io_function.get_file_list_by_ext(img_file_extension,sub_image_dir,bsub_folder=False)
        for filename in sub_image_list:
            h3_id = get_h3_id_in_sub_images_f(filename)
            sub_img_id_path_dict.setdefault(h3_id, []).append(filename)
    for h3_id in sub_img_id_path_dict.keys():
        cell_dir = os.path.join(out_dir, h3_id)
        for img_path in sub_img_id_path_dict[h3_id]:
            io_function.movefiletodir(img_path,cell_dir,b_verbose=False)

    print(datetime.now(), f'Completed organizing sub-image')

def get_set_name_from_tif(tif_path, h3_id):
    tif_name = os.path.basename(tif_path)
    return  tif_name.split(f'id{h3_id}')[0][:-1]

# copied and modified from ~/codes/PycharmProjects/rs_data_proc/DEMscripts/dem_diff_to_colorRelief.py
def dem_tif_to_colorReleif(input,output,out_format='GTiff',tif_compression='lzw'):
    # change the file extension
    file_extension = raster_io.get_file_extension(out_format)
    file_path, ext1 = os.path.splitext(output)
    output = file_path + file_extension

    if os.path.isfile(output):
        basic.outputlogMessage('%s exists, skip'%output)
        return True

    color_text_file = 'dem_diff_color_5to5m.txt'

    if out_format=='GTiff':
        command_str = f'gdaldem color-relief -of {out_format} -co compress={tif_compression} -co tiled=yes -co bigtiff=if_safer '
    else:
        command_str = f'gdaldem color-relief -of {out_format} '

    command_str +=  input + ' ' + ' %s '%color_text_file  + output

    # print(command_str)
    # res = os.system(command_str)
    return basic.os_system_exit_code(command_str)

def convert_tif_to_png(tif, save_path, set_name):

    if os.path.isfile(save_path):
        print('%s exists, skip' % save_path)
        return

    #if it elevation, need to use gdaldem
    if set_name == "samElev":
        return dem_tif_to_colorReleif(tif,save_path, out_format='PNG')


    command_str = "gdal_translate -of PNG %s %s" % (tif, save_path)
    basic.os_system_exit_code(command_str)

def convert_geojson_to_pixel_json(geojson, pixel_json, set_name, ref_image):
    pixel_json = pixel_json.replace('geojson','json')
    if os.path.isfile(pixel_json):
        print(f'Warning, {pixel_json} exists, skip')
        return
    polys = vector_gpd.read_polygons_gpd(geojson,b_fix_invalid_polygon=False)
    img_transform = raster_io.get_transform_from_file(ref_image)

    height,  width,  band_count, _ = raster_io.get_height_width_bandnum_dtype(ref_image)

    Polygons = {}
    # convert to pixel coordinates
    for idx, poly in enumerate(polys):
        # print('polygon:', poly, class_int)
        object = {}
        x, y = [], []
        if isinstance(poly, Polygon):
            x, y = poly.exterior.coords.xy
        elif isinstance(poly, MultiPolygon):
            for sub_poly in poly.geoms:  # Iterate over individual Polygons
                x_s, y_s = sub_poly.exterior.coords.xy
                x.extend(x_s)
                y.extend(y_s)
        pixel_xs, pixel_ys = raster_io.geo_xy_to_pixel_xy(x, y, img_transform)
        # print(pixel_xs,pixel_ys)

        points = [[int(xx), int(yy)] for xx, yy in zip(pixel_xs, pixel_ys)]
        object['rings'] = points
        object['index'] = idx
        object['image_set'] = set_name
        object['imageSize'] = { "width": width, "height": height }

        Polygons[f'poly_{idx}'] = object

    # io_function.save_dict_to_txt_json(pixel_json, Polygons)
    json_data = json.dumps(Polygons)
    with open(pixel_json, "w") as f_obj:
        f_obj.write(json_data)

    pass

def add_text(im, text, xy=(1, 1), font=None, font_size=10, fill=(255,255,255,255), stroke_fill=(0,0,0,255), stroke_width=1):
    # im = im.convert("RGBA") # convert to "RGBA" causeing some blur areas
    if font is None:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
    draw = ImageDraw.Draw(im)
    draw.text(xy, text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
    return im

def save_multiple_png_to_gif(png_file_list, title_list, save_path='output.gif', start_str='s2_', duration_ms=500):

    if os.path.isfile(save_path):
        print(f'GIF: {save_path} exist, skip')
        return

    # png_file_list = sorted(png_file_list)
    # Pair, sort by the png filename, then unzip
    pairs = sorted(zip(png_file_list, title_list), key=lambda x: x[0])
    png_file_list, title_list = map(list, zip(*pairs))

    # draw title on the image? No. Just select sentinel 2 images
    sel_png_file_list = []
    sel_title_list = []
    for png, title in zip(png_file_list,title_list):
        if title.startswith(start_str):   # select specific images
            # calculate entropy, ignore these with very small entropy (0.5)
            valid_per, entropy = raster_io.get_valid_percent_shannon_entropy(png, log_base=10)
            if entropy < 0.5:
                print(f'warning, entropy for {os.path.basename(png)} is {entropy}, ignore it for gif construction')
                continue
            sel_png_file_list.append(png)
            sel_title_list.append(title)
    png_file_list = sel_png_file_list
    title_list = sel_title_list
    if len(png_file_list) < 1:
        print('Warning, no PNG files')
        return

    # print(title_list)
    year_list = [get_year_in_file_or_str(item) for item in title_list]
    # print('year_list',year_list)

    # Load as RGB (opaque)
    frames_rgb = [Image.open(fn).convert("RGB") for fn in png_file_list]
    frames_rgb = [add_text(fn, title) for fn,title in zip(frames_rgb, year_list)]

    # Quantize with adaptive palette + dithering
    frames_p = [
        im.convert("P", palette=Image.ADAPTIVE, colors=256, dither=Image.FLOYDSTEINBERG)
        for im in frames_rgb
    ]

    # Save GIF without transparency/disposal
    frames_p[0].save(save_path, save_all=True,append_images=frames_p[1:],duration=duration_ms,loop=0,optimize=False)


def convert_2_web_format_one_h3_grid(h3_f,out_dir,b_rm_org_file):

    h3_id = os.path.basename(h3_f)
    h3_grid_ext = os.path.join(h3_f, 'h3_cell_' + h3_id + '.geojson')
    h3_save_dir = os.path.join(out_dir, h3_id)
    if os.path.isdir(h3_save_dir) is False:
        io_function.mkdir(h3_save_dir)
    tif_list = io_function.get_file_list_by_pattern(h3_f, '*.tif')
    png_list = []
    set_name_list = []
    for tif in tif_list:
        save_png_path = os.path.join(h3_save_dir, io_function.get_name_no_ext(tif) + '.png')
        set_name = get_set_name_from_tif(tif, h3_id)
        convert_tif_to_png(tif, save_png_path, set_name)
        if os.path.isfile(save_png_path):
            png_list.append(save_png_path)
            set_name_list.append(set_name)

        # convert the corresponding geojson
        geojson_f = os.path.join(h3_f, f'{set_name}_{h3_id}.geojson')
        if os.path.isfile(geojson_f):
            geojson_f_save = os.path.join(h3_save_dir, os.path.basename(geojson_f))
            convert_geojson_to_pixel_json(geojson_f, geojson_f_save, set_name, tif)
            if b_rm_org_file:
                io_function.delete_file_or_dir(geojson_f)

        # convert h3 grid
        h3_grid_ext_save = os.path.join(h3_save_dir,
                                        os.path.basename(io_function.get_name_by_adding_tail(h3_grid_ext, set_name)))
        convert_geojson_to_pixel_json(h3_grid_ext, h3_grid_ext_save, set_name, tif)

        if b_rm_org_file:
            io_function.delete_file_or_dir(tif)

    # save multiple png files into a gif

    gif_save_path_l7 = os.path.join(h3_save_dir, 'xGIF_' + 'id' + h3_id + '.gif')
    save_multiple_png_to_gif(png_list, set_name_list, save_path=gif_save_path_l7, start_str='l7_')  # select landsat7

    gif_save_path_l8 = os.path.join(h3_save_dir, 'yGIF_' + 'id' + h3_id + '.gif')
    save_multiple_png_to_gif(png_list, set_name_list, save_path=gif_save_path_l8, start_str='l8_')  # select landsat8

    # put "z_", making sure it on the left most after sorting
    gif_save_path_s2 = os.path.join(h3_save_dir, 'zGIF_' + 'id' + h3_id + '.gif')
    save_multiple_png_to_gif(png_list, set_name_list, save_path=gif_save_path_s2,
                             start_str='s2_')  # all s2 image, not including s2nir

    if b_rm_org_file:
        io_function.delete_file_or_dir(h3_grid_ext)

def convert_2_web_format(data_dir, out_dir, b_rm_org_file=False, process_num=1):
    h3_grid_folders = io_function.get_file_list_by_pattern(data_dir,'*')
    h3_grid_folders = [item for item in h3_grid_folders if len(os.path.basename(item))==15 ]
    if len(h3_grid_folders) < 1:
        raise IOError(f'No H3 grid folder in {data_dir}')

    if os.path.isdir(out_dir) is False:
        io_function.mkdir(out_dir)

    if process_num == 1:
        for idx, h3_f in enumerate(h3_grid_folders):
            # print progress
            if idx % 100 == 0:
                print(datetime.now(),f'( {idx+1}/{len(h3_grid_folders)})Processing {h3_f}')
            convert_2_web_format_one_h3_grid(h3_f, out_dir, b_rm_org_file)
    elif process_num > 1:
        theadPool = Pool(process_num)
        parameters_list = [(h3_f, out_dir, b_rm_org_file) for idx, h3_f in enumerate(h3_grid_folders)]
        results = theadPool.starmap(convert_2_web_format_one_h3_grid, parameters_list)
        theadPool.close()
    else:
        raise ValueError(f'Invalid process_num: {process_num}')


def test_convert_2_web_format():
    data_dir = os.path.expanduser('~/Data/rts_ArcticDEM_mapping/validation/data_multi_test')
    out_dir = os.path.join(data_dir,'png')
    convert_2_web_format(data_dir, out_dir, b_rm_org_file=False)

def test_save_multiple_png_to_gif():
    # png_dir = os.path.expanduser('~/Data/rts_ArcticDEM_mapping/validation/data_multi_png/880d68cb29fffff')

    png_dir = os.path.expanduser('~/Data/rts_ArcticDEM_mapping/validation/select_by_rf_09_v3_png/8802f04ed7fffff')
    print(f'testing using pngs in {png_dir}')
    png_list = sorted(io_function.get_file_list_by_pattern(png_dir,'*.png'))
    print('png_list', png_list)
    title_list = [os.path.basename(item)  for item in png_list]
    print('title_list', title_list)
    # start_str = 's2_' # s2 images, not including s2nir
    start_str = 'l7_' # l7 images
    # start_str = 'l8_' # l8 images

    save_multiple_png_to_gif(png_list,title_list,save_path='output.gif', start_str=start_str)

def main(options, args):
    grid_path = args[0]
    out_dir = options.out_dir
    buffer_size= options.buffer_size
    process_num= options.process_num
    b_extract_subImg_only = options.b_extract_subImg_only

    mapping_res_ini = options.mapping_res_ini
    if mapping_res_ini is None:
        print('Please set "--mapping_res_ini"')
        return

    print('b_extract_subImg_only:', b_extract_subImg_only)

    t0 = time.time()
    grid_gpd = gpd.read_file(grid_path)
    t1 = time.time()
    print(f'Loaded grid vector file, containing {len(grid_gpd)} cells, {len(grid_gpd.columns)} columns, cost {t1-t0} seconds')
    column_names = grid_gpd.columns.to_list()
    # remove "_A" (area) and "_C" (count), only keep these columns name end with "_A" or "_C"
    column_pre_names = [item.replace('_A','') for item in column_names if "_A" in item ]
    # column_pre_names = [item.replace('_C','') for item in column_pre_names]
    basic.outputlogMessage(f'column names: {column_names}')
    basic.outputlogMessage(f'column_pre_names: {column_pre_names}')

    column_pre_names_ini = sorted(list(set([item.split('-')[0] for item in parameters.get_Parameter_names(mapping_res_ini)])))
    basic.outputlogMessage(f'column_pre_names_ini: {column_pre_names_ini}')
    if len(column_pre_names) != len(column_pre_names_ini):
        basic.outputlogMessage('Info, The count in "column_names" and "column_pre_names" is different, some data was not used in mapping,'
                               'will use the column in "column_pre_names"')
        column_pre_names = column_pre_names_ini


    mapping_shp_raster_dict = get_mapping_shp_raster_dict(column_pre_names, mapping_res_ini)
    io_function.save_dict_to_txt_json('mapping_shp_raster_dict.json',mapping_shp_raster_dict)

    if os.path.isdir(out_dir) is False:
        io_function.mkdir(out_dir)

    obtain_multi_data(grid_gpd,grid_path, mapping_shp_raster_dict, out_dir, buffersize=buffer_size,
                      process_num=process_num,b_get_subImg_only=b_extract_subImg_only)

    png_dir = out_dir + '_png' #os.path.join(out_dir,'png')
    convert_2_web_format(out_dir, png_dir, b_rm_org_file=False,process_num=process_num)



if __name__ == '__main__':
    # test_convert_2_web_format()
    # test_save_multiple_png_to_gif()
    # sys.exit(0)

    usage = "usage: %prog [options] grid_vector "
    parser = OptionParser(usage=usage, version="1.0 2025-9-4")
    parser.description = 'Introduction: extract multiple images and vector for each grid (cell) for validation '

    parser.add_option("-o", "--out_dir",
                      action="store", dest="out_dir",default="data_multi",
                      help="the output directory ")

    parser.add_option("-i", "--mapping_res_ini",
                      action="store", dest="mapping_res_ini",
                      help="the ini file contain mapping results and the corresponding rasters")

    parser.add_option("-b", "--buffer_size",
                      action="store", dest="buffer_size", type=float, default='50',
                      help="the buffer size (in meters) for extracting sub-images")

    parser.add_option("-p", "--process_num",
                      action="store", dest="process_num",type=int, default=4,
                      help="the process number for extracting sub-images")

    parser.add_option("-e", "--b_extract_subImg_only",
                      action="store_true", dest="b_extract_subImg_only",default=False,
                      help="if set, will only extract sub-images, not organizing them into h3 id folder and getting polygons")

    # parser.add_option("-b", "--using_bounding_box",
    #                   action="store_true", dest="using_bounding_box",default=False,
    #                   help="whether use the boudning boxes of polygons, this can avoid some invalid"
    #                        " polygons and be consistent with YOLO output")




    (options, args) = parser.parse_args()
    # print(options.no_label_image)
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)
