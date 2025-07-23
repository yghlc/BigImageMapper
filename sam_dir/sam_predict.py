#!/usr/bin/env python
# Filename: sam_predict.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 15 June 2023
"""

import os, sys
import time
from datetime import datetime
from optparse import OptionParser

import pandas as pd

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters
import basic_src.io_function as io_function
import basic_src.basic as basic
import basic_src.map_projection as map_projection
import datasets.split_image as split_image
import datasets.raster_io as raster_io
import datasets.vector_gpd as vector_gpd

import GPUtil
from multiprocessing import Process

import numpy as np
import torch
# import torchvision

from pycocotools import mask as mask_utils

def is_file_exist_in_folder(folder):
    # just check if the folder is empty
    if len(os.listdir(folder)) == 0:
        return False
    else:
        return True

def copy_one_patch_image_data(patch, entire_img_data):
    #(xoff,yoff ,xsize, ysize)
    row_s = patch[1]
    row_e = patch[1] + patch[3]
    col_s = patch[0]
    col_e = patch[0] + patch[2]
    # entire_img_data is in opencv format:  height, width, band_num
    patch_data = entire_img_data[row_s:row_e, col_s:col_e, :]
    # print(patch_data.shape) # e.g., (1171, 1000, 1), ndim = 3
    if entire_img_data.shape[2] == 1:
        # duplicate to three bands
        patch_data = np.repeat(patch_data[:, :], 3, axis=2)
    # print(patch_data.shape)
    return patch_data

def save_masks_as_shape(patch_boundary, masks,ref_raster, save_path, min_area=None, max_area=None, b_prompt=False):
    # covert masks to polygons, then save as vector files directly
    if len(masks) < 0:
        print('Warning, no masks')
        return False

    if b_prompt:
        all_polygons = masks['mask']
        all_values = masks['value']
    else:
        all_polygons = []
        all_values = []
        # everything mode
        for idx, mask in enumerate(masks):
            mask_array = mask_utils.decode(mask["segmentation"])
            geometry_list, raster_values = raster_io.numpy_array_to_shape(mask_array,ref_raster,boundary=patch_boundary,nodata=0,connect8=True)
            all_polygons.extend(geometry_list)
            all_values.extend(raster_values)

    # save to disk
    if len(all_polygons) > 0:
        all_polygons_shapely = [ vector_gpd.json_geometry_to_polygons(item) for item in all_polygons ]
        # remove some small and big one
        small_idx = []
        if min_area is not None:
            small_idx = [ idx for idx, poly in enumerate(all_polygons_shapely) if poly.area < min_area]
        big_idx = []
        if max_area is not None:
            big_idx = [ idx for idx, poly in enumerate(all_polygons_shapely) if poly.area > max_area]
        small_idx.extend(big_idx)
        if len(small_idx) > 0:
            all_polygons_shapely = [item for idx, item in enumerate(all_polygons_shapely) if idx not in small_idx]
            all_values = [item for idx, item in enumerate(all_values) if idx not in small_idx]

        save_path = save_path.replace('.tif','.gpkg')
        # return +init=epsg:3413, got FutureWarning: '+init=<authority>:<code>' syntax is deprecated. '<authority>:<code>' is the preferred initialization method
        # prj4 = raster_io.get_projection(ref_raster,format='proj4') #
        wkt_str = raster_io.get_projection(ref_raster, format='wkt')
        # 'area': [item.area for item in all_polygons_shapely]     # no need to save area
        data_pd = pd.DataFrame({'polygon':all_polygons_shapely, 'DN': all_values} )
        vector_gpd.save_polygons_to_files(data_pd,'polygon',wkt_str,save_path, format='GPKG')



def save_masks_to_disk(accumulate_count, patch_boundary, masks,ref_raster, save_path,scores=None, b_prompt=False):
    # patch boundary: (xoff,yoff ,xsize, ysize)
    # Set output image data type based on the number of objects
    if len(masks) < 0:
        print('Warning, no masks')
        return False

    if b_prompt:
        # dtype = np.uint8
        # best_mask = masks[scores.argmax(),:,:]
        # # seg_map = np.zeros((best_mask.shape[0],best_mask.shape[1]), dtype=dtype)
        # seg_map = best_mask.astype(dtype)
        seg_map = masks
    else:
        # everything mode
        dtype = np.uint32
        mask0_array = mask_utils.decode(masks[0]["segmentation"])
        seg_map = np.zeros((mask0_array.shape[0],mask0_array.shape[1]), dtype=dtype)
        for idx, mask in enumerate(masks):
            mask_array = mask_utils.decode(mask["segmentation"])
            # print(accumulate_count + idx + 1)
            # print(mask_array.shape)
            # print(np.count_nonzero(mask_array))
            seg_map[mask_array != 0] = accumulate_count + idx + 1
        seg_map[seg_map == 0] = accumulate_count

    raster_io.save_numpy_array_to_rasterfile(seg_map,save_path,ref_raster,compress='lzw', tiled='yes', bigtiff='if_safer',
                                             boundary=patch_boundary,verbose=False)


def get_prompt_points_list(prompts_path, image_path):
    # convert points to x,y
    points, attribute_values = vector_gpd.read_polygons_attributes_list(prompts_path,['class_int','poly_id'],b_fix_invalid_polygon=False)
    if len(points) < 1:
        return [], [], []
    if points[0].geom_type != 'Point':
        raise ValueError('The geometry type should be Point, not %s'%str(points[0].geom_type))

    # points to pixel coordinates
    x_list = [item.x for item in points]
    y_list = [item.y for item in points]
    img_transform = raster_io.get_transform_from_file(image_path)
    cols, rows = raster_io.geo_xy_to_pixel_xy(x_list, y_list, img_transform)
    points_pixel_list = [ [x, y] for x,y in zip(cols, rows)]
    # save to txt
    # save_point_txt = os.path.splitext(io_function.get_name_by_adding_tail(prompts_path,'pixel'))[0] + '.txt'
    # io_function.save_list_to_txt(save_point_txt,points_pixel_list)
    # print(points_pixel_list)
    # print(class_values)
    class_values = attribute_values[0]
    poly_ids = attribute_values[1]

    return points_pixel_list, class_values, poly_ids

def get_prompt_boxes_list(prompts_path, image_path):
    # convert points to x,y
    boxes, attribute_values = vector_gpd.read_polygons_attributes_list(prompts_path,['class_int','poly_id'],b_fix_invalid_polygon=False)
    if len(boxes) < 1:
        return [], [], []
    if boxes[0].geom_type != 'Polygon':
        raise ValueError('The geometry type should be Polygon, not %s'%str(boxes[0].geom_type))

    ## get (x1,y1, x2, y2) list. (left, up, right, down)
    xyxy_geo_list = [vector_gpd.get_box_polygon_leftUp_rightDown(item) for item in boxes]
    x1_list = []
    y1_list = []
    x2_list = []
    y2_list = []
    for x1,y1,x2,y2 in xyxy_geo_list:
        x1_list.append(x1)
        y1_list.append(y1)
        x2_list.append(x2)
        y2_list.append(y2)

    # points to pixel coordinates
    img_transform = raster_io.get_transform_from_file(image_path)
    cols1, rows1 = raster_io.geo_xy_to_pixel_xy(x1_list, y1_list, img_transform)
    cols2, rows2 = raster_io.geo_xy_to_pixel_xy(x2_list, y2_list, img_transform)

    xyxy_pixel_list = [ [xp1, yp1, xp2, yp2] for xp1, yp1, xp2, yp2 in zip(cols1, rows1, cols2, rows2)]
    class_values = attribute_values[0]
    poly_ids = attribute_values[1]

    # save to txt
    # save_point_txt = os.path.splitext(io_function.get_name_by_adding_tail(prompts_path,'pixel'))[0] + '.txt'
    # io_function.save_list_to_txt(save_point_txt,xyxy_pixel_list)
    # print(xyxy_pixel_list)
    # print(class_values)

    return xyxy_pixel_list, class_values, poly_ids


def is_a_point_within_patch(point, bounds):
    # bounds (xoff,yoff ,xsize, ysize)
    if point[0] > bounds[0] and point[0] < bounds[0]+bounds[2] and point[1] > bounds[1] and point[1] < bounds[1]+bounds[3]:
        return True
    return False

def is_a_box_totally_within_a_path(box, bounds):
    # bounds (xoff,yoff ,xsize, ysize)
    # box: (x1,y1, x2, y2), i.e. (left, up, right, down)
    if box[0] >= bounds[0] and box[2] < bounds[0]+bounds[2] and box[1] >= bounds[1] and box[3] < bounds[1]+bounds[3]:
        return True
    return False

def get_prompt_points_a_patch(points_pixel_list, class_values, group_ids, patch_boundary):
    # extract points in a patch
    # patch boundary: (xoff,yoff ,xsize, ysize)
    idx_list = [idx for idx, p in enumerate(points_pixel_list) if is_a_point_within_patch(p,patch_boundary)]
    points_sel = [[points_pixel_list[idx][0] - patch_boundary[0],
                   points_pixel_list[idx][1] - patch_boundary[1] ] for idx in idx_list]     # substract xoff, yoff
    class_values_sel = [class_values[idx] for idx in idx_list]
    group_ids_sel = [group_ids[idx] for idx in idx_list]
    return points_sel, class_values_sel, group_ids_sel

def get_prompt_boxes_a_patch(boxes_pixel_list, class_values, group_ids, patch_boundary, b_ignore_touch_edge=True):
    # extract boxes in a patch
    # box list: (x1,y1, x2, y2) list. (left, up, right, down)
    # patch boundary: (xoff,yoff ,xsize, ysize)
    # when b_ignore_touch_edge is true, then ignore those boxes touch the patch edge

    if b_ignore_touch_edge:
        idx_list = [idx for idx, box in enumerate(boxes_pixel_list) if is_a_box_totally_within_a_path(box,patch_boundary)]
        box_sel = [ [boxes_pixel_list[idx][0] - patch_boundary[0],
                     boxes_pixel_list[idx][1] - patch_boundary[1],
                     boxes_pixel_list[idx][2] - patch_boundary[0],
                     boxes_pixel_list[idx][3] - patch_boundary[1]]  for idx in idx_list ]   # substract xoff, yoff
        class_values_sel = [class_values[idx] for idx in idx_list]
        group_ids_sel = [group_ids[idx] for idx in idx_list]
        return box_sel, class_values_sel, group_ids_sel
    else:
        raise ValueError('not support yet')


def group_prompt_points_boxes(points_pixel_list, class_values, group_ids,input_boxes, box_label, box_group_id):
    # group_points = {}
    # group_classes = {}
    # for pt, c_v, g_id in zip(points_pixel_list, class_values, group_ids):
    #     group_points.setdefault(g_id,[]).append(pt)
    #     group_classes.setdefault(g_id,[]).append(c_v)
    # return group_points, group_classes

    group_prompts_all = {}
    if points_pixel_list is not None:
        for pt, c_v, g_id in zip(points_pixel_list, class_values, group_ids):
            group_prompts_all.setdefault(g_id,{}).setdefault('point',[]).append(pt)
            group_prompts_all.setdefault(g_id,{}).setdefault('p_class',[]).append(c_v)
    if input_boxes is not None:
        for box, c_v, g_id in zip(input_boxes, box_label, box_group_id):
            group_prompts_all.setdefault(g_id, {}).setdefault('box', []).append(box)
            group_prompts_all.setdefault(g_id, {}).setdefault('b_class', []).append(c_v)

    return group_prompts_all

def segment_rs_image_sam(image_path, save_dir, model, model_type, patch_w, patch_h, overlay_x, overlay_y,
                        batch_size=1, min_area=10, max_area=40000, prompts=None, finetune_m=None,sam_version='1'):

    # for each region, after SAM, its area (in pixel) should be within [min_area, max_area],
    # otherwise, remove it

    # sam_version: "1" or sam, "2" for "sam2"
    if sam_version == '1':
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    elif sam_version == '2':
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    else:
        raise ValueError(f'unknown on support sam version: {sam_version}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if finetune_m is None:
        basic.outputlogMessage('loading a checkpoint: %s'%model)
        if sam_version == '1':
            sam = sam_model_registry[model_type](checkpoint=model)
        elif sam_version == '2':
            sam = build_sam2(model_type, model, device=device)
        else:
            pass
    else:
        # load the trained model
        from fine_tune_sam import ModelSAM
        model_trained = ModelSAM()
        model_trained.setup(model_type, model)
        basic.outputlogMessage('loading fine-tuned model: %s'%finetune_m)
        model_trained.load_state_dict(torch.load(finetune_m, map_location=torch.device(device)) )
        sam = model_trained.model

    sam.to(device=device)
    # if torch.cuda.is_available():
    #     sam.to(device='cuda')
    # else:
    #     sam.to(device='cpu')

    if prompts is not None and isinstance(prompts, list) is False:
        prompts = [prompts]
    prompts_dict = {}

    # points_pixel, class_values, group_ids = None, None, None
    # boxes_pixel, box_class_values, box_group_ids = None, None, None
    # print('Debug, prompts is:', prompts)
    if prompts is None:
        if overlay_x > 0 or overlay_y >0:
            raise ValueError('For everything mode, overlay_x and overlay_y should be zero')
        # segment everything
        if sam_version=='1':
            mask_generator = SamAutomaticMaskGenerator(sam, output_mode="coco_rle")  # "binary_mask" take a lot of memory
        elif sam_version == '2':
            mask_generator = SAM2AutomaticMaskGenerator(sam, output_mode="coco_rle")
        else:
            pass
    else:
        # only segment targets
        if sam_version == '1':
            mask_generator = SamPredictor(sam)
        elif sam_version == '2':
            mask_generator = SAM2ImagePredictor(sam)
        else:
            pass


        # read prompts
        for p_path in prompts:
            if p_path.endswith('point.shp'):
                points_pixel, class_values, group_ids = get_prompt_points_list(p_path,image_path)
                prompts_dict['point'] = {'xy_pixel':points_pixel,'class_value':class_values, 'group_id':group_ids }
            elif p_path.endswith('box.shp'):
                boxes_pixel, box_class_values, box_group_ids = get_prompt_boxes_list(p_path, image_path)
                prompts_dict['box'] = {'xyxy_pixel': boxes_pixel, 'class_value': box_class_values, 'group_id': box_group_ids}
            else:
                raise ValueError('Cannot find prompt type in the file name: %s'%os.path.basename(p_path))

        # points_pixel, class_values, group_ids = get_prompt_points_list(prompts,image_path)
        # prompts_box = io_function.get_name_by_adding_tail(prompts,'box')
        # print('prompts_box:', prompts_box)
        # if os.path.isfile(prompts_box):
        #     boxes_pixel, box_class_values, box_group_ids = get_prompt_boxes_list(prompts_box, image_path)

    height, width, band_num, date_type = raster_io.get_height_width_bandnum_dtype(image_path)
    # print('input image: height, width, band_num, date_type',height, width, band_num, date_type)
    xres, yres = raster_io.get_xres_yres_file(image_path)

    b_use_memory = True
    if height* width > 50000*50000:  # 10000*10000 is a threshold, can be changed
        b_use_memory = False

    # read the entire image
    if b_use_memory:
        entire_img_data, nodata = raster_io.read_raster_all_bands_np(image_path)
        entire_img_data = entire_img_data.transpose(1, 2, 0)  # to opencv format  # in HWC uint8 format
        # # # RGB to BGR: Matplotlib image to OpenCV https://www.scivision.dev/numpy-image-bgr-to-rgb/
        # entire_img_data = entire_img_data[..., ::-1].copy() # no need, hlc July 9, 2023. in amg.py, they use cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        entire_height, entire_width, band_num = entire_img_data.shape
        print("entire_height, entire_width, band_num", entire_height, entire_width, band_num)
    if band_num not in [1, 3]:
        raise ValueError('only accept one band or three band images')

    if os.path.isdir(save_dir) is False:
        io_function.mkdir(save_dir)
    # write a file there to indicate the segmentation has started
    with open(os.path.join(save_dir, 'started.txt'), 'w') as f_obj:
        f_obj.writelines(str(datetime.now()) + ': The process has started\n')

    # divide the image the many small patches, then calculate one by one, solving memory issues.
    image_patches = split_image.sliding_window(width, height, patch_w, patch_h, adj_overlay_x=overlay_x,
                                               adj_overlay_y=overlay_y)
    # patch boundary: (xoff,yoff ,xsize, ysize)
    patch_count = len(image_patches)
    total_seg_count = 0

    for p_idx, a_patch in enumerate(image_patches):
        t0 = time.time()
        # get width, height, and band_num of a patch, then create a darknet image.
        if b_use_memory:
            image = copy_one_patch_image_data(a_patch, entire_img_data)
        else:
            image, nodata = raster_io.read_raster_all_bands_np(image_path, boundary=a_patch)
            image = image.transpose(1, 2, 0)
        # height, width, band_num = img_data.shape
        # if band_num not in [1, 3]:
        #     raise ValueError('only accept one band or three band images')
        # print('Debug, image.shape',image.shape)
        if np.std(image[:,:, 0]) < 0.0001:
            print(f'Patch ({p_idx}/{patch_count}), shape: {image[:,:, 0].shape} is black or white, ignore')
            continue

        # save file name
        file_name = "I%d_%d" % (0, p_idx)
        save_path = os.path.join(save_dir, file_name + '.tif')

        if prompts is None:
            masks = mask_generator.generate(image)
            masks = [item for item in masks  if item['area'] >= min_area and item['area'] <= max_area]    # remove big and small region
            # save_masks_to_disk(total_seg_count,a_patch,masks, image_path,save_path)
            save_masks_as_shape(a_patch,masks,image_path,save_path, min_area=min_area*(xres**2))
            total_seg_count += len(masks)
        else:
            # generate masks based on input points
            # pause here, August 16, 2023
            input_point, input_label, group_id = None, None, None
            if 'point' in prompts_dict.keys():
                input_point, input_label, group_id = get_prompt_points_a_patch( prompts_dict['point']['xy_pixel'],
                                                                                prompts_dict['point']['class_value'],
                                                                                prompts_dict['point']['group_id'], a_patch)
            input_boxes, box_label, box_group_id = None, None, None
            if 'box' in prompts_dict.keys():
                input_boxes, box_label, box_group_id= get_prompt_boxes_a_patch(prompts_dict['box']['xyxy_pixel'],
                                                                                prompts_dict['box']['class_value'],
                                                                                prompts_dict['box']['group_id'], a_patch)

            # input_point = np.array(input_point[-2:-1])
            # input_label = np.array(input_label[-2:-1])
            # input_point = np.array(input_point[:10])
            # input_label = np.array([1 for i in range(10)])
            # print(input_point)
            # print(input_label)
            if len(input_point) < 1:
                continue

            group_prompts_dict = group_prompt_points_boxes(input_point, input_label, group_id,input_boxes, box_label, box_group_id)
            mask_generator.set_image(image)
            # seg_map = np.zeros((a_patch[3], a_patch[2]), dtype=np.uint32)
            seg_map_results = {'mask':[], 'value':[]}
            for key_id in group_prompts_dict.keys():
                a_group_prompt = group_prompts_dict[key_id]
                points = np.array(a_group_prompt['point']) if 'point' in a_group_prompt.keys() else None
                p_labels = np.array(a_group_prompt['p_class']) if 'p_class' in a_group_prompt.keys() else None
                boxes = np.array(a_group_prompt['box']) if 'box' in a_group_prompt.keys() else None
                b_labels = np.array(a_group_prompt['b_class']) if 'b_class' in a_group_prompt.keys() else None

                # for each group, usually, only have one box, but may have multiple points
                if np.sum(b_labels) == 0:   # 0 is background, we don't need boxes for background
                    boxes = None

                # if all the labels are 0 (background), then ignore this group
                if p_labels is not None and np.any(p_labels==1) is False:
                    basic.outputlogMessage('warning, In group %d, all points are labeled as 0, ignore this group'%key_id)
                    continue
                # print(key_id, points, labels)
                b_multimask = False
                if points is not None and len(points) < 2:
                    b_multimask = True
                # b_multimask = True if len(points) < 2 and boxes is None else False
                # for the case only use box, b_multimask is also False in the example.
                masks, scores, logits = mask_generator.predict(
                    point_coords=points,
                    point_labels=p_labels,
                    box = boxes,
                    multimask_output=b_multimask,
                )
                # get the best segment map
                group_seg_map = masks[scores.argmax(),:,:]  # the output is True or False

                # calculate area, remove mask that is too small or too big
                seg_map_size_pixel = int(np.sum(group_seg_map))

                # print('type', type(min_area), type(max_area), type(seg_map_size_pixel))
                if seg_map_size_pixel < min_area or seg_map_size_pixel > max_area:
                    # print('removed, %d'%key_id)
                    continue
                # print('key_id: %d, its seg map has %s pixel, (min, max are %s, %s)  ' % (key_id, str(seg_map_size_pixel), min_area, max_area))
                # print('polygon count: %d'%len(seg_map_results['mask']))

                # print(group_seg_map.dtype, group_seg_map.shape)
                # seg_map[group_seg_map ] = key_id   # save key id
                group_seg_map = group_seg_map.astype(np.uint8)
                # print('group_seg_map uint8, pixel count: ', np.sum(group_seg_map))

                # after numpy_array_to_shape, one region may end in several polygons, and some of them are very small.
                # in the later step (save_masks_as_shape), remove these tiny polygons
                geometry_list, raster_values = raster_io.numpy_array_to_shape(group_seg_map, image_path,
                                                                              boundary=a_patch, nodata=0,
                                                                              connect8=True)
                seg_map_results['mask'].extend(geometry_list)
                seg_map_results['value'].extend([key_id]*len(geometry_list))

            # save to disk
            # save_masks_to_disk(0, a_patch, seg_map, image_path, save_path, scores=None, b_prompt=True)
            save_masks_as_shape(a_patch, seg_map_results, image_path, save_path, min_area=min_area*(xres**2), b_prompt=True)



            # id = 0
            # for pnt, lab in zip(input_point,input_label):
            #     print(pnt, lab)
            #     masks, scores, logits = mask_generator.predict(
            #         point_coords=np.array([pnt]),
            #         point_labels=np.array([lab]),
            #         multimask_output=True,
            #     )
            #     #                     return_logits = True,
            #     print(masks.shape, np.max(masks), np.min(masks))
            #     print(scores)
            #     save_path2 = io_function.get_name_by_adding_tail(save_path,'%d'%id)
            #     save_masks_to_disk(0,a_patch,masks,image_path, save_path2, scores=scores, b_prompt=True)
            #     id += 1
            # break # for test


        # if p_idx % 100 == 0:
        print('Processed %d patch, total: %d, this batch costs %f second' % (p_idx, patch_count, time.time() - t0))


def get_prompts_for_an_image(image_path, area_prompt_path, save_dir,prompt_type='point'):
    '''
    extract prompts (points or boxes), specific for this image
    :param area_prompt_path:
    :param save_dir:
    :param prompt_type: point or box
    :return:
    '''
    if area_prompt_path is None:
        return None
    if os.path.isdir(save_dir) is False:
        io_function.mkdir(save_dir)
    prompt_path = os.path.join(save_dir, io_function.get_name_no_ext(image_path) + '_prompts_%s.shp'%prompt_type)
    if os.path.isfile(prompt_path):
        basic.outputlogMessage('%s exists, skip extracting prompts for this image'%prompt_path)
        return prompt_path

    t0 = time.time()

    ## get prompts, specific for this image
    #TODO: need to exclude no data regions
    img_bounds = raster_io.get_image_bound_box(image_path)
    # gpd.clip is very slow when the input vector file is very large
    # img_prj = map_projection.get_raster_or_vector_srs_info_proj4(image_path)
    # prompt_path = vector_gpd.clip_geometries(area_prompt_path,prompt_path,img_bounds, target_prj=img_prj)

    # use ogr2ogr to crop
    prompt_path = vector_gpd.clip_geometries_ogr2ogr(area_prompt_path, prompt_path, img_bounds, format='ESRI Shapefile')
    print('crop shapefile: %s, costs %f second' % (area_prompt_path, time.time() - t0))
    return prompt_path

def segment_remoteSensing_image(para_file, area_ini, image_path, save_dir, network_ini, batch_size=1):
    '''
    segment
    :param para_file:
    :param image_path:
    :param save_dir:
    :param network_ini:
    :param batch_size:
    :return:
    '''

    patch_w = parameters.get_digit_parameters(para_file, "inf_patch_width", 'int')
    patch_h = parameters.get_digit_parameters(para_file, "inf_patch_height", 'int')
    overlay_x = parameters.get_digit_parameters(para_file, "inf_pixel_overlay_x", 'int')
    overlay_y = parameters.get_digit_parameters(para_file, "inf_pixel_overlay_y", 'int')

    # sam_mask_min_area = parameters.get_digit_parameters(para_file, "sam_mask_min_area_pixel", 'int')
    # sam_mask_max_area = parameters.get_digit_parameters(para_file, "sam_mask_max_area_pixel", 'int')
    xres, yres = raster_io.get_xres_yres_file(image_path)
    sam_mask_min_area = parameters.get_digit_parameters(para_file, "sam_mask_min_area_m2", 'float')
    sam_mask_max_area = parameters.get_digit_parameters(para_file, "sam_mask_max_area_m2", 'float')
    sam_mask_min_area = sam_mask_min_area/(xres**2)    # to pixels
    sam_mask_max_area = sam_mask_max_area/(xres**2)

    model = parameters.get_file_path_parameters(network_ini,'checkpoint')
    sam_version = parameters.get_string_parameters(network_ini, 'sam_version')
    if sam_version == '1':
        model_type = parameters.get_string_parameters(network_ini,'model_type')
    elif sam_version == '2':
        model_type = parameters.get_file_path_parameters(network_ini, 'model_type') # it's the config file
    else:
        raise ValueError(f'unknown on support sam version: {sam_version}')

    finedtuned_model = parameters.get_file_path_parameters_None_if_absence(network_ini,'finedtuned_model')

    # prepare prompts (points or boxes)
    prompt_type = parameters.get_string_parameters_None_if_absence(para_file, 'prompt_type')
    # print('Debug, prompt_type', prompt_type)

    if prompt_type is None:
        prompts_an_image_list = None
    else:
        prompt_type_list = prompt_type.split('+')
        prompt_path = parameters.get_file_path_parameters(area_ini, 'prompt_path')

        if prompt_path.endswith('.txt'):
            prompts_list = io_function.read_list_from_txt(prompt_path)
            prompts_list = [os.path.join(os.path.dirname(prompt_path), item)  for item in prompts_list]
        else:
            prompts_list = [prompt_path]
        if 'No-Prompts' in prompts_list[0]:
            basic.outputlogMessage(f'Warning, No-Prompts for {area_ini}, skip')
            return

        # in the case when each prompt type has multiple (>1) vector files, making things complicated.
        # so, when generating prompts for each region, merge these prompts into one file
        # checking if one type have only one prompt file
        if len(prompt_type_list) != len(prompts_list):
            raise ValueError('for each region, each prompt type should only have one vector file')

        # only keep those match the prompt type
        prompts_list_new = []
        for p_type in prompt_type_list:
            for tmp in prompts_list:
                if tmp.endswith('%s.shp'%p_type):
                    prompts_list_new.append(tmp)
        prompts_list = prompts_list_new

        prompts_an_image_list = []
        for p_path in prompts_list:
            #TODO: after crop, the number of points and boxes may not the same
            if p_path.endswith('point.shp'):
                prompt_image_path = get_prompts_for_an_image(image_path, p_path,save_dir,prompt_type='point')
            elif p_path.endswith('box.shp'):
                prompt_image_path = get_prompts_for_an_image(image_path, p_path, save_dir, prompt_type='box')
            else:
                raise ValueError('Cannot find prompt type in the file name: %s'%os.path.basename(p_path))
            if prompt_image_path is None:
                continue
            prompts_an_image_list.append(prompt_image_path)

    # using the python API
    out = segment_rs_image_sam(image_path, save_dir, model, model_type,
                               patch_w, patch_h, overlay_x, overlay_y, batch_size=batch_size,
                               min_area=sam_mask_min_area, max_area=sam_mask_max_area,
                               prompts=prompts_an_image_list,finetune_m=finedtuned_model,
                               sam_version=sam_version)

def segment_one_image_sam(para_file, area_ini, image_path, img_save_dir, inf_list_file, gpuid):

    network_ini = parameters.get_string_parameters(para_file, 'network_setting_ini')
    inf_batch_size = parameters.get_digit_parameters(para_file, 'inf_batch_size', 'int')

    done_indicator = '%s_done' % inf_list_file
    if os.path.isfile(done_indicator):
        basic.outputlogMessage('warning, %s exist, skip prediction' % done_indicator)
        return
    # use a specific GPU for prediction, only inference one image
    time0 = time.time()
    if gpuid is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)

    segment_remoteSensing_image(para_file, area_ini, image_path, img_save_dir, network_ini, batch_size=inf_batch_size)

    duration = time.time() - time0
    os.system('echo "$(date): time cost of segmenting an image in %s: %.2f seconds">>"time_cost.txt"' % (
    inf_list_file, duration))
    # write a file to indicate that the prediction has done.
    os.system('echo %s > %s_done' % (inf_list_file, inf_list_file))
    return

def parallel_segment_main(para_file):
    print("Segment Anything (run parallel if using multiple GPUs)")
    machine_name = os.uname()[1]
    start_time = datetime.now()

    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    outdir = parameters.get_directory(para_file, 'inf_output_dir')
    # remove previous results (let user remove this folder manually or in exe.sh folder)
    io_function.mkdir(outdir)

    # get name of inference areas
    multi_inf_regions = parameters.get_string_list_parameters(para_file, 'inference_regions')
    b_use_multiGPUs = parameters.get_bool_parameters(para_file, 'b_use_multiGPUs')
    maximum_prediction_jobs = parameters.get_digit_parameters(para_file, 'maximum_prediction_jobs', 'int')

    # loop each inference regions
    sub_tasks = []
    for area_idx, area_ini in enumerate(multi_inf_regions):
        basic.outputlogMessage(f'({area_idx+1}/{len(multi_inf_regions)}) working on {area_ini}')

        area_name = parameters.get_string_parameters(area_ini, 'area_name')
        area_remark = parameters.get_string_parameters(area_ini, 'area_remark')
        area_time = parameters.get_string_parameters(area_ini, 'area_time')

        inf_image_dir = parameters.get_directory(area_ini, 'inf_image_dir')

        # it is ok consider a file name as pattern and pass it the following functions to get file list
        inf_image_or_pattern = parameters.get_string_parameters(area_ini, 'inf_image_or_pattern')

        inf_img_list = io_function.get_file_list_by_pattern(inf_image_dir, inf_image_or_pattern)
        img_count = len(inf_img_list)
        if img_count < 1:
            raise ValueError(
                'No image for inference, please check inf_image_dir (%s) and inf_image_or_pattern (%s) in %s'
                % (inf_image_dir, inf_image_or_pattern, area_ini))

        area_save_dir = os.path.join(outdir, area_name + '_' + area_remark + '_' + area_time)
        io_function.mkdir(area_save_dir)

        # parallel inference images for this area
        CUDA_VISIBLE_DEVICES = []
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            CUDA_VISIBLE_DEVICES = [int(item.strip()) for item in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        idx = 0
        while idx < img_count:

            while basic.alive_process_count(sub_tasks) >= maximum_prediction_jobs:
                print(datetime.now(),
                      '%d jobs are running simultaneously, wait 5 seconds' % basic.alive_process_count(sub_tasks))
                time.sleep(60)  # wait 60 seconds, then check the count of running jobs again

            if b_use_multiGPUs:
                # get available GPUs  # https://github.com/anderskm/gputil
                # memory: orders the available GPU device ids by ascending memory usage
                deviceIDs = GPUtil.getAvailable(order='memory', limit=100, maxLoad=0.5,
                                                maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
                # only use the one in CUDA_VISIBLE_DEVICES
                if len(CUDA_VISIBLE_DEVICES) > 0:
                    deviceIDs = [item for item in deviceIDs if item in CUDA_VISIBLE_DEVICES]
                    basic.outputlogMessage('on ' + machine_name + ', available GPUs:' + str(deviceIDs) +
                                           ', among visible ones:' + str(CUDA_VISIBLE_DEVICES))
                else:
                    basic.outputlogMessage('on ' + machine_name + ', available GPUs:' + str(deviceIDs))

                if len(deviceIDs) < 1:
                    time.sleep(60)  # wait 60 seconds, then check the available GPUs again
                    continue
                # set only the first available visible
                gpuid = deviceIDs[0]
                basic.outputlogMessage(
                    '%d: predict image %s on GPU %d of %s' % (idx, inf_img_list[idx], gpuid, machine_name))
            else:
                gpuid = None
                basic.outputlogMessage('%d: predict image %s on %s' % (idx, inf_img_list[idx], machine_name))

            # run inference
            img_save_dir = os.path.join(area_save_dir, 'I%d' % idx)
            inf_list_file = os.path.join(area_save_dir, '%d.txt' % idx)

            done_indicator = '%s_done' % inf_list_file
            if os.path.isfile(done_indicator):
                basic.outputlogMessage('warning, %s exist, skip prediction' % done_indicator)
                idx += 1
                continue

            # if it already exists, then skip
            if os.path.isdir(img_save_dir) and is_file_exist_in_folder(img_save_dir):
                basic.outputlogMessage('folder of %dth image (%s) already exist, '
                                       'it has been predicted or is being predicted' % (idx, inf_img_list[idx]))
                idx += 1
                continue

            with open(inf_list_file, 'w') as inf_obj:
                inf_obj.writelines(inf_img_list[idx] + '\n')

            sub_process = Process(target=segment_one_image_sam,
                                  args=(para_file, area_ini, inf_img_list[idx], img_save_dir, inf_list_file, gpuid))

            sub_process.start()
            sub_tasks.append(sub_process)

            if b_use_multiGPUs is False:
                # wait until previous one finished
                while sub_process.is_alive():
                    time.sleep(1)

            idx += 1

            # wait until predicted image patches exist or exceed 20 minutes
            time0 = time.time()
            elapsed_time = time.time() - time0
            while elapsed_time < 20 * 60:
                elapsed_time = time.time() - time0
                file_exist = os.path.isdir(img_save_dir) and is_file_exist_in_folder(img_save_dir)
                if file_exist is True or sub_process.is_alive() is False:
                    break
                else:
                    time.sleep(1)

            if sub_process.exitcode is not None and sub_process.exitcode != 0:
                sys.exit(1)

            basic.close_remove_completed_process(sub_tasks)
            # if 'chpc' in machine_name:
            #     time.sleep(60)  # wait 60 second on ITSC services
            # else:
            #     time.sleep(10)
            time.sleep(15) # wait a few seconds allowing models being loaded before moving to next

    # check all the tasks already finished
    wait_all_finish = 0
    while basic.b_all_process_finish(sub_tasks) is False:
        if wait_all_finish % 100 == 0:
            basic.outputlogMessage('wait all tasks to finish')
        time.sleep(1)
        wait_all_finish += 1

    basic.close_remove_completed_process(sub_tasks)
    end_time = datetime.now()

    diff_time = end_time - start_time
    out_str = "%s: time cost of total parallel inference on %s: %d seconds" % (
        str(end_time), machine_name, diff_time.total_seconds())
    basic.outputlogMessage(out_str)
    with open("time_cost.txt", 'a') as t_obj:
        t_obj.writelines(out_str + '\n')

def main(options, args):

    para_file = args[0]
    parallel_segment_main(para_file)

if __name__ == '__main__':
    usage = "usage: %prog [options] para_file"
    parser = OptionParser(usage=usage, version="1.0 2023-06-15")
    parser.description = 'Introduction: run segmentation using segment anything model '

    # parser.add_option("-m", "--trained_model",
    #                   action="store", dest="trained_model",
    #                   help="the trained model for prediction")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)
