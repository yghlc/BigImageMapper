#!/usr/bin/env python
# Filename: objDet_utils.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 20 August, 2025
"""


import os,sys

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import basic_src.io_function as  io_function
import basic_src.basic as basic
import datasets.vector_gpd as vector_gpd
import datasets.raster_io as raster_io
import parameters

import rasterio
import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def convert_geometry_bounding_boxes_to_YOLO(img_path, vector_path, class_column="class_int", class_adjust=-1):
    """
    Converts vector geometries in geospatial coordinates to YOLO bounding boxes in pixel coordinates,
    using a specified class index column.

    Args:
        img_path (str): Path to geospatial raster image (GeoTIFF).
        vector_path (str): Path to vector file (Shapefile, GeoJSON, etc.).
        class_column (str): Name of the column with class indices.
        class_adjust: -1, the classes were orignally set for semantic segmentation: 0 for background, while in YOLO,
        0 is the first classes

    Returns:
        list of str: YOLO annotation lines (class x_center y_center w h, all normalized to [0,1]).
    """
    # 1. Open raster to get transform and image size
    with rasterio.open(img_path) as src:
        transform = src.transform
        width = src.width
        height = src.height
        img_crs = src.crs

    # 2. Read vector data and reproject to match raster
    gdf = gpd.read_file(vector_path)
    if gdf.crs != img_crs:  # give a warning or error, this should be checkout outside this function
        raise ValueError(f'Map projection inconsistency between image: {img_path} and vector {vector_path}')
        # gdf = gdf.to_crs(img_crs)

    yolo_boxes = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        # Use the specified class column, default to 0 if missing/NaN
        class_idx = row.get(class_column, 0)
        class_idx = int(class_idx) + class_adjust

        # Get bounds in geospatial coordinates
        minx, miny, maxx, maxy = geom.bounds

        # Convert geospatial coordinates to pixel coordinates
        px_minx, px_miny = ~transform * (minx, miny)
        px_maxx, px_maxy = ~transform * (maxx, maxy)

        x1, x2 = sorted([px_minx, px_maxx])
        y1, y2 = sorted([px_miny, px_maxy])

        # Compute YOLO format (normalized: x_center, y_center, width, height)
        x_center = (x1 + x2) / 2.0 / width
        y_center = (y1 + y2) / 2.0 / height
        bbox_width = abs(x2 - x1) / width
        bbox_height = abs(y2 - y1) / height

        yolo_line = f"{class_idx} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
        yolo_boxes.append(yolo_line)

    return yolo_boxes

def plot_yolo_bboxes(tif_path,txt_path,class_names=None, box_color='red',save_path=None):
    """
    Plot TIFF image with YOLO bounding boxes, optionally save to disk.

    Args:
        tif_path (str): Path to the .tif image.
        txt_path (str): Path to the YOLO .txt file.
        class_names (list or None): Optional, list of class names.
        box_color (str): Color for bounding boxes.
        save_path (str or None): If provided, path to save the figure.
    """
    # Read the image using rasterio
    with rasterio.open(tif_path) as src:
        image = src.read()
        # Convert to HWC for matplotlib if needed
        if image.shape[0] == 1:
            image = np.squeeze(image, axis=0)  # Grayscale
        else:
            image = np.transpose(image, (1, 2, 0))  # (bands, h, w) -> (h, w, bands)
        h, w = image.shape[:2]

    # Read YOLO bboxes
    bboxes = []
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, x_center, y_center, bw, bh = map(float, parts)
                bboxes.append((int(class_id), x_center, y_center, bw, bh))
    except FileNotFoundError:
        print(f"YOLO label file not found: {txt_path}")
        bboxes = []

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    if image.ndim == 2:
        ax.imshow(image, cmap='gray')
    else:
        if image.shape[2] > 3:
            image = image[:, :, :3]  # Only plot first 3 bands
        ax.imshow(image.astype(np.uint8))

    for bbox in bboxes:
        class_id, x_center, y_center, bw, bh = bbox
        # Convert normalized coordinates to pixel values
        x = (x_center - bw/2) * w
        y = (y_center - bh/2) * h
        box_w = bw * w
        box_h = bh * h
        rect = patches.Rectangle((x, y), box_w, box_h, linewidth=2, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect)
        # Label
        if class_names:
            label = class_names[class_id]
        else:
            label = str(class_id)
        ax.text(x, y-5, label, color=box_color, fontsize=12, weight='bold', backgroundcolor='white')

    ax.set_axis_off()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        plt.close(fig)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

def checking_yolo_boxes():
    class_names = ['RTS']
    # class_names = None
    data_dir = os.path.expanduser('~/Data/rts_ArcticDEM_mapping/object_detection_s2/training_data/train_set01_v1_s2_rgb_2024/subImages')

    file_names = ['s2_2024_RGB_id0095800934_1326_48', 's2_2024_RGB_id0111100748_1412_35', 's2_2024_RGB_id0064700509_1084-from_2_grids_22',
                  's2_2024_RGB_id0043500570_713-from_3_grids_18','s2_2024_RGB_id0030100681_161_3','s2_2024_RGB_id0041900584_613-from_3_grids_12']

    for file_name in file_names:
        img_path = os.path.join(data_dir,f'{file_name}.tif')
        txt_path = os.path.join(data_dir,f'{file_name}.txt')
        save_path = f'{file_name}_plot_v2.jpg'
        plot_yolo_bboxes(img_path,txt_path,class_names=class_names, save_path=save_path)



def get_bounding_boxes_from_vector_file(img_path_list, vector_path, b_ignore_edge_objects=False):
    # the img_path_list are those subImages, generated by "get_subImages.py", using the "vector_path"

    if vector_gpd.is_field_name_in_shp(vector_path, 'class_int') is False:
        raise ValueError(f'Attribute: class_int is not in {vector_path}')

    save_box_txt_list = []

    for idx, img_path in enumerate(img_path_list):
        # crop the vector file
        raster_bounds = raster_io.get_image_bound_box(img_path) # bounding box: (left, bottom, right, top)
        img_res_x, _ = raster_io.get_xres_yres_file(img_path)
        file_path_no_ext = os.path.splitext(img_path)[0]
        save_crop_vector = file_path_no_ext + '.gpkg'
        save_box_txt = file_path_no_ext + '.txt'
        if not os.path.isfile(save_crop_vector):
            out_path = vector_gpd.clip_geometries(vector_path,save_crop_vector,raster_bounds, format='GPKG')
            if out_path is None:
                # basic.outputlogMessage(f'Warning, No polygons or boxes for {os.path.relpath(img_path)}')

                # if yolo_boxes is empty, this would save an empty file needed by YOLO
                io_function.save_list_to_txt(save_box_txt, [])
                save_box_txt_list.append(save_box_txt)
                continue

            #keep or not keep these polygons touches the edge, if not keep, rise warning
            if b_ignore_edge_objects:
                # io_function.copy_file_to_dst(save_crop_vector, io_function.get_name_by_adding_tail(save_crop_vector,'bak'))
                vector_gpd.remove_polygon_boxes_touch_edge(save_crop_vector,raster_bounds, shrink_meters=img_res_x)


        if not os.path.isfile(save_box_txt):
            yolo_boxes = convert_geometry_bounding_boxes_to_YOLO(img_path,save_crop_vector)
            # if yolo_boxes is empty, this would save an empty file needed by YOLO
            io_function.save_list_to_txt(save_box_txt,yolo_boxes)

        save_box_txt_list.append(save_box_txt)

    return save_box_txt_list


def get_file_list(input_dir, pattern, area_ini):
    file_list = io_function.get_file_list_by_pattern(input_dir, pattern)
    if len(file_list) < 1:
        raise ValueError('No files for processing, please check directory (%s) and pattern (%s) in %s'
                         % (input_dir, pattern, area_ini))
    return file_list

def get_merged_training_data_txt(training_data_dir, expr_name, region_count):
    save_path = os.path.join(training_data_dir,
                             'merge_training_data_for_%s_from_%d_regions.txt' % (expr_name, region_count))
    return save_path

def read_image_list_from_txt(txt_path):
    lines = io_function.read_list_from_txt(txt_path)
    image_path_list = [ item.split(':')[0] for item in lines]
    return image_path_list

def save_training_data_to_yolo_format_darknet(para_file, train_sample_txt, val_sample_txt):
    # copied and modified from "yolov4_dir/pre_yolo_data.py"
    # write obj.data file

    train_img_list = read_image_list_from_txt(train_sample_txt)
    val_img_list = read_image_list_from_txt(val_sample_txt)


    expr_name = parameters.get_string_parameters(para_file,'expr_name')
    num_classes_noBG = parameters.get_digit_parameters(para_file, 'NUM_CLASSES_noBG', 'int')
    object_names = parameters.get_string_list_parameters(para_file,'object_names')
    io_function.mkdir('data')
    io_function.mkdir(expr_name)

    with open(os.path.join('data','obj.data'), 'w') as f_obj:
        f_obj.writelines('classes = %d'%num_classes_noBG + '\n')

        train_txt = os.path.join('data','train.txt')
        io_function.save_list_to_txt(train_txt,train_img_list)
        f_obj.writelines('train = %s'%train_txt+ '\n')

        val_txt = os.path.join('data','val.txt')
        io_function.save_list_to_txt(val_txt, val_img_list)
        f_obj.writelines('valid = %s' % val_txt + '\n')

        obj_name_txt = os.path.join('data','obj.names')
        io_function.save_list_to_txt(obj_name_txt,object_names)
        f_obj.writelines('names = %s' % obj_name_txt + '\n')

        f_obj.writelines('backup = %s'%expr_name + '\n')
    pass


def main():
    checking_yolo_boxes()
    pass


if __name__ == '__main__':
    main()
