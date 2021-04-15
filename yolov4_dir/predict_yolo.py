#!/usr/bin/env python
# Filename: predict_yolo 
"""
introduction: running prediction using yolov4

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 07 April, 2021
"""

import os, sys
import time
from datetime import datetime
from optparse import OptionParser

import GPUtil
from multiprocessing import Process

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters

import basic_src.io_function as io_function
import basic_src.basic as basic
import datasets.split_image as split_image
import datasets.raster_io as raster_io
import datasets.build_RS_data as build_RS_data

# add darknet Python API
darknet_dir = os.environ.get('DARKNET_PATH', './')
sys.path.insert(0, darknet_dir)
import darknet
import cv2
import numpy as np
import json

def is_file_exist_in_folder(folder):
    # only check the first ten files
    # update on July 21, 2020. For some case, the first 10 may not exist (ignore if they are black)
    # so, if we find any file exist from 0 to 100000, then return True
    # file_ext = ['.json', '.png','.jpg','.tif']
    # for i in range(100000):
    #     for ext in file_ext:
    #         if os.path.isfile(os.path.join(folder, '%d'%i + ext)):
    #             return True
    # return False

    # just check if the folder is empty
    if len(os.listdir(folder)) == 0:
        return False
    else:
        return True
    # file_list = io_function.get_file_list_by_pattern(folder, '*.*')  # this may take time if a lot of file exist
    # if len(file_list) > 0:
    #     return True
    # else:
    #     return False

def b_all_task_finish(all_tasks):
    for task in all_tasks:
        if task.is_alive():
            return False
    return True

def split_an_image(para_file, image_path,save_dir, patch_w, patch_h, overlay_x, overlay_y):

    split_format = parameters.get_string_parameters(para_file, 'split_image_format')
    out_format = 'PNG'  # default is PNG
    if split_format == '.tif': out_format = 'GTIFF'
    if split_format == '.jpg': out_format = 'JPEG'
    if os.path.isdir(save_dir) is False:
        io_function.mkdir(save_dir)

    split_image.split_image(image_path, save_dir, patch_w, patch_h, overlay_x, overlay_y, out_format, pre_name=None,process_num=8)
    # get list
    patch_list = io_function.get_file_list_by_ext(split_format,save_dir,bsub_folder=False)
    if len(patch_list) < 1:
        print('Wanring, no images in %s'%save_dir)
        return None
    list_txt_path = save_dir + '_list.txt'
    io_function.save_list_to_txt(list_txt_path,patch_list)
    return list_txt_path

def load_darknet_network(config_file,data_file, weights, batch_size = 1):
    network, class_names, class_colors = darknet.load_network(
        config_file,
        data_file,
        weights,
        batch_size=batch_size
    )
    return network, class_names, class_colors

def darknet_image_detection_v0(image_path, network, class_names, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect

    # no need to use network width and size, only if images have the same size
    # width = darknet.network_width(network)
    # height = darknet.network_height(network)

    image = cv2.imread(image_path)

    height, width, _ = image.shape
    darknet_image = darknet.make_image(width, height, 3)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image_resized = cv2.resize(image_rgb, (width, height),
    #                            interpolation=cv2.INTER_LINEAR)

    print('image_rgb shape', image_rgb.shape)
    for ii in range(3):
        band = image_rgb[:,:,ii]
        print('band %d'%ii, 'mean: %f'%np.mean(band))
    # using rasterio to read, then check if it's the same as cv2.
    img_data, nodata = raster_io.read_raster_all_bands_np(image_path)
    print('img_data shape', img_data.shape)
    img_data = img_data.transpose(1,2,0)
    print('img_data shape', img_data.shape)
    for ii in range(3):
        band = img_data[:,:,ii]
        print('band %d'%ii, 'mean: %f'%np.mean(band))

    # darknet.copy_image_from_bytes(darknet_image, image_rgb.tobytes())
    darknet.copy_image_from_bytes(darknet_image, img_data.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)

    darknet.free_image(darknet_image)

    return detections

def darknet_image_detection(image_path, network, class_names, thresh):
    # Create one with image we reuse for each detect : images with the same size.
    img_data, nodata = raster_io.read_raster_all_bands_np(image_path)
    img_data = img_data.transpose(1, 2, 0)
    height, width, band_num = img_data.shape
    if band_num not in [1,3]:
        raise ValueError('only accept one band or three band images')
    darknet_image = darknet.make_image(width, height, band_num)

    # darknet.copy_image_from_bytes(darknet_image, image_rgb.tobytes())
    darknet.copy_image_from_bytes(darknet_image, img_data.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)

    return detections

# def test_darknet_image_detection():
#     # run in ~/Data/Arctic/canada_arctic/autoMapping/multiArea_yolov4_1
#     # run inside Vagrant machine and call singularity container.
#     print('\n')
#     print('Run test_darknet_image_detection')
#
#     config_file = 'yolov4_obj.cfg'
#     yolo_data = os.path.join('data','obj.data')
#     weights = os.path.join('exp1', 'yolov4_obj_best.weights')
#
#     img_path = os.path.join('debug_img', '20200818_mosaic_8bit_rgb_0_class_1_p_0.png')
#
#     network, class_names, _ = load_darknet_network(config_file, yolo_data, weights, batch_size=1)
#     # detections = darknet_image_detection_v0(img_path, network, class_names, 0.1)
#     detections = darknet_image_detection(img_path, network, class_names, 0.1)
#     for label, confidence, bbox in detections:
#         bbox = darknet.bbox2points(bbox)    # to [xmin, ymin, xmax, ymax]
#         print(label, class_names.index(label), bbox, confidence)
#         # print('%s %s %s %s'%(str(label), str(class_names.index(label)), str(bbox), str(confidence)))
#         # x, y, w, h = convert2relative(image, bbox)
#         # label = class_names.index(label)
#         # f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))

def check_batch_shape(images, batch_size):
    """
        Image sizes should be the same width and height
    """
    shapes = [image.shape for image in images]
    if len(set(shapes)) > 1:
        raise ValueError("Images don't have same shape")
    if len(shapes) > batch_size:
        raise ValueError("Batch size higher than number of images")
    return shapes[0]

def prepare_batch(images, network, channels=3):
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    darknet_images = []
    for image in images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        custom_image = image_resized.transpose(2, 0, 1)
        darknet_images.append(custom_image)

    batch_array = np.concatenate(darknet_images, axis=0)
    batch_array = np.ascontiguousarray(batch_array.flat, dtype=np.float32)/255.0
    darknet_images = batch_array.ctypes.data_as(darknet.POINTER(darknet.c_float))
    return darknet.IMAGE(width, height, channels, darknet_images)

def darknet_batch_detection(network, images, class_names,class_colors,
                    thresh=0.25, hier_thresh=.5, nms=.45, batch_size=4):
    # hier_thresh: hierarchical softmax
    image_height, image_width, _ = check_batch_shape(images, batch_size)
    print(image_height, image_width)
    darknet_images = prepare_batch(images, network)
    print("done prepare_batch")
    batch_detections = darknet.network_predict_batch(network, darknet_images, batch_size, image_width,
                                                     image_height, thresh, hier_thresh, None, 0, 0)
    print("done prediction")
    batch_predictions = []
    for idx in range(batch_size):
        num = batch_detections[idx].num
        detections = batch_detections[idx].dets
        if nms:
            darknet.do_nms_obj(detections, num, len(class_names), nms)
        predictions = darknet.remove_negatives(detections, class_names, num)
        images[idx] = darknet.draw_boxes(predictions, images[idx], class_colors)
        batch_predictions.append(predictions)
    darknet.free_batch_detections(batch_detections, batch_size)
    return images, batch_predictions


# def test_darknet_batch_images_detection():
#     # run in ~/Data/Arctic/canada_arctic/autoMapping/multiArea_yolov4_1
#     # run inside Vagrant machine and call singularity container.
#     print('\n')
#     print('Run test_darknet_batch_images_detection')
#
#     config_file = 'yolov4_obj.cfg'
#     yolo_data = os.path.join('data','obj.data')
#     weights = os.path.join('exp1', 'yolov4_obj_best.weights')
#     batch_size = 3
#
#     # these three have the same size
#     image_names = ['20200818_mosaic_8bit_rgb_p_1001.png', '20200818_mosaic_8bit_rgb_p_1038.png',
#                    '20200818_mosaic_8bit_rgb_p_1131.png']
#     image_names = [os.path.join('debug_img', item)  for item in image_names]
#
#     # load network
#     network, class_names, class_colors = load_darknet_network(config_file, yolo_data, weights, batch_size=batch_size)
#
#     images = [cv2.imread(image) for image in image_names]
#     for cv_img, img_path in zip(images,image_names):
#         print(img_path, cv_img.shape)
#     images, detections = darknet_batch_detection(network, images, class_names, class_colors, batch_size=batch_size)
#
#
#     for img, det in zip(image_names,detections):
#         print(img)
#         for label, confidence, bbox in det:
#             bbox = darknet.bbox2points(bbox)    # to [xmin, ymin, xmax, ymax]
#             print(label, class_names.index(label), bbox, confidence)
#
#     for name, image in zip(image_names, images):
#         cv2.imwrite(name.replace("20200818", "res_20200818"), image)
#     # for label, confidence, bbox in detections:
#     #     bbox = darknet.bbox2points(bbox)    # to [xmin, ymin, xmax, ymax]
#     #     print(label, class_names.index(label), bbox, confidence)


def copy_one_patch_image_data(patch, entire_img_data):
    #(xoff,yoff ,xsize, ysize)
    row_s = patch[1]
    row_e = patch[1] + patch[3]
    col_s = patch[0]
    col_e = patch[0] + patch[2]
    # entire_img_data is in opencv format:  height, width, band_num
    patch_data = entire_img_data[row_s:row_e, col_s:col_e, :]
    return patch_data


def darknet_batch_detection_rs_images(network, image_path,save_dir, patch_groups, patch_count, class_names,batch_size,
                    thresh=0.25, hier_thresh=.5, nms=.45):
    '''
    # run batch detection of YOLO on an remote sensing image.
    :param network: a darknet network (already load the weight)
    :param image_path: the image patch.
    :param patch_groups: the group of images patch, each group has the same width and height.
    :param class_names:
    :param batch_size:
    :param thresh:
    :param hier_thresh:
    :param nms:
    :return:
    '''
    # read the entire image
    entire_img_data, nodata = raster_io.read_raster_all_bands_np(image_path)
    entire_img_data = entire_img_data.transpose(1, 2, 0)    # to opencv format
    entire_height, entire_width, band_num = entire_img_data.shape
    print("entire_height, entire_width, band_num",entire_height, entire_width, band_num)
    if band_num not in [1, 3]:
        raise ValueError('only accept one band or three band images')

    # class_colors={} #{'thaw_slump': (139, 140, 228)}  # class_colors should get when load network
    # for name in class_names:
    #     class_colors[name] = (0,0,0)

    patch_idx = 0
    for key in patch_groups.keys():
        patches_sameSize = patch_groups[key]

        batch_patches = [patches_sameSize[i*batch_size : (i+1)*batch_size] for i in
                 range((len(patches_sameSize) + batch_size - 1) // batch_size )]

        for a_batch_patch in batch_patches:
            t0 = time.time()
            input_batch_size = len(a_batch_patch)
            if input_batch_size != batch_size:
                if input_batch_size > batch_size:
                    raise ValueError('input_batch_size (%d) is greater than batch_size (%d)'%(input_batch_size,batch_size))
                while(len(a_batch_patch)<batch_size):
                    a_batch_patch.append(a_batch_patch[0])  # copy the first one to fit the batch size
            images = [copy_one_patch_image_data(patch,entire_img_data) for patch in a_batch_patch ]

            # for the batch prediction, it seems have to resize to the network size
            img_height, img_width, band_num = images[0].shape
            width = darknet.network_width(network)
            height = darknet.network_height(network)        

            # darknet_images = prepare_batch(images, network)
            # prepare_batch
            darknet_images = []
            for image in images:
                # print('image shape',image.shape)
                image_resized = cv2.resize(image, (width, height),interpolation=cv2.INTER_LINEAR)
                if band_num == 1:
                    image_resized = np.expand_dims(image_resized, axis=2) # add one diminsion at the last
                # print('image_resized shape', image_resized.shape)
                custom_image = image_resized.transpose(2, 0, 1)
                darknet_images.append(custom_image)
            batch_array = np.concatenate(darknet_images, axis=0)
            batch_array = np.ascontiguousarray(batch_array.flat, dtype=np.float32) / 255.0
            darknet_images = batch_array.ctypes.data_as(darknet.POINTER(darknet.c_float))
            darknet_images = darknet.IMAGE(width, height, band_num, darknet_images)

            # prediction
            batch_detections = darknet.network_predict_batch(network, darknet_images, batch_size, img_height,
                                                             img_width, thresh, hier_thresh, None, 0, 0)
            batch_predictions = []
            for idx in range(input_batch_size):
                num = batch_detections[idx].num
                detections = batch_detections[idx].dets
                if nms:
                    darknet.do_nms_obj(detections, num, len(class_names), nms)
                predictions = darknet.remove_negatives(detections, class_names, num)
                # images[idx] = np.ascontiguousarray(images[idx])
                # images[idx] = darknet.draw_boxes(predictions, images[idx], class_colors)
                batch_predictions.append(predictions)
            darknet.free_batch_detections(batch_detections, batch_size)

            for idx, (patch,image,predictions) in enumerate(zip(a_batch_patch,images,batch_predictions)):
                # remove the duplicated ones
                if idx >= input_batch_size:
                    break
                # save results
                save_res_json = os.path.join(save_dir, '%d.json' % patch_idx)
                # the confidence output by network_predict_batch is 0-1, convert to percent
                save_one_patch_detection_json(patch, predictions, class_names, save_res_json, b_percent=True)

                
                # cv2.imwrite('%d.png'%patch_idx, image)    # save for testing
                # for label, confidence, bbox in predictions:
                #     bbox = darknet.bbox2points(bbox)  # to [xmin, ymin, xmax, ymax]
                #     print(label, class_names.index(label), bbox, confidence)

                if patch_idx % 100 == 0:
                    print('saving %d patch, total: %d, cost %f second' % (patch_idx, patch_count, (time.time() - t0)/batch_size))

                patch_idx += 1


def test_darknet_batch_detection_rs_images():
    print('\n')
    print('Run test_darknet_batch_detection_rs_images')

    config_file = 'yolov4_obj_oneband.cfg'
    yolo_data = os.path.join('data', 'obj.data')
    weights = os.path.join('exp1', 'yolov4_obj_oneband_best.weights')
    batch_size = 1

    # these three have the same size
    # image_names = ['20200818_mosaic_8bit_rgb_p_1001.png'] #, '20200818_mosaic_8bit_rgb_p_1038.png', '20200818_mosaic_8bit_rgb_p_1131.png']
    image_names = ['Alaska_north_slope_hillshade_20170426_poly_28_p_0.png'] #, '20200818_mosaic_8bit_rgb_p_1038.png', '20200818_mosaic_8bit_rgb_p_1131.png']
    image_names = [os.path.join('debug_img', item) for item in image_names]
    image_path = image_names[0]
    save_dir = './'

    height, width, band_num, date_type = raster_io.get_height_width_bandnum_dtype(image_path)
    # print('input image: height, width, band_num, date_type',height, width, band_num, date_type)

    # divide the image the many small patches, then calcuate one by one, solving memory issues.
    patch_w = 480
    patch_h = 480
    overlay_x = 160
    overlay_y = 160
    image_patches = split_image.sliding_window(width,height,patch_w,patch_h,adj_overlay_x=overlay_x,adj_overlay_y=overlay_y)
    patch_count = len(image_patches)

    if os.path.isdir(save_dir) is False:
        io_function.mkdir(save_dir)

    # group patches based on size, each patch is (xoff,yoff ,xsize, ysize)
    patch_groups = {}
    for patch in image_patches:
        wh_str = 'w%d'%patch[2] + '_' + 'h%d'%patch[3]
        if wh_str in patch_groups.keys():
            patch_groups[wh_str].append(patch)
        else:
            patch_groups[wh_str] = [patch]


    # load network
    network, class_names, class_colors = load_darknet_network(config_file, yolo_data, weights, batch_size=batch_size)

    darknet_batch_detection_rs_images(network, image_path, save_dir, patch_groups, patch_count, class_names, batch_size,
                                      thresh=0.25, hier_thresh=.5, nms=.45)



def save_one_patch_detection_json(patch,detections,class_names,save_res_json,b_percent=False):
    objects = []
    for label, confidence, bbox in detections:
        bbox = darknet.bbox2points(bbox)  # to [xmin, ymin, xmax, ymax]
        bbox = [bbox[0] + patch[0], bbox[1] + patch[1], bbox[2] + patch[0],
                bbox[3] + patch[1]]  # to entire image coordinate

        if b_percent:
            confidence = round(confidence*100,2)
        object = {'class_id': class_names.index(label),
                  'name': label,
                  'bbox': bbox,
                  'confidence': confidence}
        objects.append(object)

    json_data = json.dumps(objects, indent=2)
    with open(save_res_json, "w") as f_obj:
        f_obj.write(json_data)


def predict_rs_image_yolo_poythonAPI(image_path, save_dir, model, config_file, yolo_data,
                                     patch_w, patch_h, overlay_x, overlay_y, batch_size=1):
    '''
    predict an remote sensing using YOLO Python API
    :param image_path:
    :param save_dir:
    :param model:
    :param config_file:
    :param yolo_data:
    :param patch_w:
    :param patch_h:
    :param overlay_x:
    :param overlay_y:
    :param batch_size:
    :return:
    '''
    height, width, band_num, date_type = raster_io.get_height_width_bandnum_dtype(image_path)
    # print('input image: height, width, band_num, date_type',height, width, band_num, date_type)

    # divide the image the many small patches, then calcuate one by one, solving memory issues.
    image_patches = split_image.sliding_window(width,height,patch_w,patch_h,adj_overlay_x=overlay_x,adj_overlay_y=overlay_y)
    patch_count = len(image_patches)

    if os.path.isdir(save_dir) is False:
        io_function.mkdir(save_dir)

    # group patches based on size, each patch is (xoff,yoff ,xsize, ysize)
    patch_groups = {}
    for patch in image_patches:
        wh_str = 'w%d'%patch[2] + '_' + 'h%d'%patch[3]
        if wh_str in patch_groups.keys():
            patch_groups[wh_str].append(patch)
        else:
            patch_groups[wh_str] = [patch]

    network, class_names, _ = load_darknet_network(config_file, yolo_data, model, batch_size=batch_size)

    # batch detection
    if batch_size > 1:
        return darknet_batch_detection_rs_images(network, image_path,save_dir, patch_groups, patch_count,class_names,batch_size)


    # read the entire image
    entire_img_data, nodata = raster_io.read_raster_all_bands_np(image_path)
    entire_img_data = entire_img_data.transpose(1, 2, 0)    # to opencv format
    entire_height, entire_width, band_num = entire_img_data.shape
    print("entire_height, entire_width, band_num",entire_height, entire_width, band_num)
    if band_num not in [1, 3]:
        raise ValueError('only accept one band or three band images')

    patch_idx = 0
    for key in patch_groups.keys():
        patches_sameSize = patch_groups[key]

        # get width, height, and band_num of a patch, then create a darknet image.
        # img_data, nodata = raster_io.read_raster_all_bands_np(image_path, boundary=patches_sameSize[0])
        # img_data = img_data.transpose(1, 2, 0)
        # height, width, band_num = img_data.shape
        # if band_num not in [1, 3]:
        #     raise ValueError('only accept one band or three band images')

        height, width = patches_sameSize[0][3], patches_sameSize[0][2]

        # Create one with image we reuse for each detect : images with the same size.
        darknet_image = darknet.make_image(width, height, band_num)

        for idx, patch in enumerate(patches_sameSize):
            t0 = time.time()
            # patch: (xoff,yoff ,xsize, ysize)
            # img_data, nodata = raster_io.read_raster_all_bands_np(image_path,boundary=patch)
            # img_data = img_data.transpose(1, 2, 0)
            img_data = copy_one_patch_image_data(patch, entire_img_data)

            # prediction
            darknet.copy_image_from_bytes(darknet_image, img_data.tobytes())
            # thresh=0.25, relative low, set to 0 will output too many, post-process later
            detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.25)

            # save results
            save_res_json = os.path.join(save_dir,'%d.json'%patch_idx)
            save_one_patch_detection_json(patch,detections,class_names,save_res_json)

            if patch_idx % 100 == 0:
                print('saving %d patch, total: %d, cost %f second'%(patch_idx,patch_count, time.time()-t0))

            patch_idx += 1

        darknet.free_image(darknet_image)






def predict_remoteSensing_image(para_file, image_path, save_dir,model, config_file, yolo_data, batch_size=1, b_python_api=True):
    '''
    run prediction of a remote sensing using yolov4
    :param image_path:
    :param model:
    :param config_file:
    :param yolo_data:
    :param batch_size:
    :param b_python_api: if true, use the python API of yolo
    :return:
    '''

    patch_w = parameters.get_digit_parameters(para_file, "inf_patch_width", 'int')
    patch_h = parameters.get_digit_parameters(para_file, "inf_patch_height", 'int')
    overlay_x = parameters.get_digit_parameters(para_file, "inf_pixel_overlay_x", 'int')
    overlay_y = parameters.get_digit_parameters(para_file, "inf_pixel_overlay_y", 'int')

    if b_python_api:
        # using the python API
        predict_rs_image_yolo_poythonAPI(image_path, save_dir, model, config_file, yolo_data,
                                         patch_w, patch_h, overlay_x, overlay_y, batch_size=batch_size)
    else:
        # divide image the many patches, then run prediction.
        patch_list_txt = split_an_image(para_file,image_path,save_dir,patch_w,patch_h,overlay_x,overlay_y)
        if patch_list_txt is None:
            return False
        result_json = save_dir + '_result.json'
        commond_str = 'darknet detector test ' + yolo_data + ' ' + config_file + ' ' + model + ' -dont_show '
        commond_str += ' -ext_output -out ' + result_json + ' < ' + patch_list_txt
        print(commond_str)
        res = os.system(commond_str)
        if res !=0:
            sys.exit(1)


def predict_one_image_yolo(para_file, image_path, img_save_dir, inf_list_file, gpuid,trained_model):

    config_file =  parameters.get_string_parameters(para_file, 'network_setting_ini')  # 'yolov4_obj.cfg'
    yolo_data = os.path.join('data','obj.data')
    # b_python_api = False
    inf_batch_size = parameters.get_digit_parameters(para_file,'inf_batch_size','int')
    b_python_api = parameters.get_bool_parameters(para_file,'b_inf_use_python_api')

    done_indicator = '%s_done'%inf_list_file
    if os.path.isfile(done_indicator):
        basic.outputlogMessage('warning, %s exist, skip prediction'%done_indicator)
        return
    # use a specific GPU for prediction, only inference one image
    time0 = time.time()
    if gpuid is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)

    predict_remoteSensing_image(para_file,image_path, img_save_dir,trained_model, config_file, yolo_data, batch_size=inf_batch_size, b_python_api=b_python_api)

    duration = time.time() - time0
    os.system('echo "$(date): time cost of inference for image in %s: %.2f seconds">>"time_cost.txt"' % (inf_list_file, duration))
    # write a file to indicate that the prediction has done.
    os.system('echo %s > %s_done'%(inf_list_file,inf_list_file))
    return


def parallel_prediction_main(para_file, trained_model):

    print("YOLO prediction using the trained model (run parallel if use multiple GPUs)")
    machine_name = os.uname()[1]
    start_time = datetime.now()

    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    expr_name = parameters.get_string_parameters(para_file, 'expr_name')
    network_cfg = parameters.get_string_parameters(para_file,'network_setting_ini')
    if trained_model is None:
        cfg_name = os.path.splitext(os.path.basename(network_cfg))[0]
        trained_model = os.path.join(expr_name, cfg_name+'_best.weights')

    outdir = parameters.get_directory(para_file, 'inf_output_dir')
    # remove previous results (let user remove this folder manually or in exe.sh folder)
    io_function.mkdir(outdir)

    # get name of inference areas
    multi_inf_regions = parameters.get_string_list_parameters(para_file, 'inference_regions')
    b_use_multiGPUs = parameters.get_bool_parameters(para_file, 'b_use_multiGPUs')

    # loop each inference regions
    sub_tasks = []
    for area_idx, area_ini in enumerate(multi_inf_regions):

        area_name = parameters.get_string_parameters(area_ini, 'area_name')
        area_remark = parameters.get_string_parameters(area_ini, 'area_remark')
        area_time = parameters.get_string_parameters(area_ini, 'area_time')

        inf_image_dir = parameters.get_directory(area_ini, 'inf_image_dir')

        # it is ok consider a file name as pattern and pass it the following functions to get file list
        inf_image_or_pattern = parameters.get_string_parameters(area_ini, 'inf_image_or_pattern')

        inf_img_list = io_function.get_file_list_by_pattern(inf_image_dir, inf_image_or_pattern)
        img_count = len(inf_img_list)
        if img_count < 1:
            raise ValueError('No image for inference, please check inf_image_dir and inf_image_or_pattern in %s' % area_ini)

        area_save_dir = os.path.join(outdir, area_name + '_' + area_remark + '_' + area_time)
        io_function.mkdir(area_save_dir)

        # parallel inference images for this area
        CUDA_VISIBLE_DEVICES = []
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            CUDA_VISIBLE_DEVICES = [int(item.strip()) for item in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        idx = 0
        while idx < img_count:

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
                    time.sleep(5)  # wait 5 seconds, then check the available GPUs again
                    continue
                # set only the first available visible
                gpuid = deviceIDs[0]
                basic.outputlogMessage('%d: predict image %s on GPU %d of %s' % (idx, inf_img_list[idx], gpuid, machine_name))
            else:
                gpuid = None
                basic.outputlogMessage('%d: predict image %s on %s' % (idx, inf_img_list[idx], machine_name))

            # run inference
            img_save_dir = os.path.join(area_save_dir, 'I%d' % idx)
            inf_list_file = os.path.join(area_save_dir, '%d.txt' % idx)

            # if it already exist, then skip
            if os.path.isdir(img_save_dir) and is_file_exist_in_folder(img_save_dir):
                basic.outputlogMessage('folder of %dth image (%s) already exist, '
                                       'it has been predicted or is being predicted' % (idx, inf_img_list[idx]))
                idx += 1
                continue

            with open(inf_list_file, 'w') as inf_obj:
                inf_obj.writelines(inf_img_list[idx] + '\n')

            sub_process = Process(target=predict_one_image_yolo,
                                  args=(para_file,inf_img_list[idx], img_save_dir, inf_list_file,
                                        gpuid, trained_model))
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

            # if 'chpc' in machine_name:
            #     time.sleep(60)  # wait 60 second on ITSC services
            # else:
            #     time.sleep(10)

    # check all the tasks already finished
    wait_all_finish = 0
    while b_all_task_finish(sub_tasks) is False:
        if wait_all_finish % 100 == 0:
            basic.outputlogMessage('wait all tasks to finish')
        time.sleep(1)
        wait_all_finish += 1

    end_time = datetime.now()

    diff_time = end_time - start_time
    out_str = "%s: time cost of total parallel inference on %s: %d seconds" % (
    str(end_time), machine_name, diff_time.seconds)
    basic.outputlogMessage(out_str)
    with open("time_cost.txt", 'a') as t_obj:
        t_obj.writelines(out_str + '\n')



def main(options, args):

    para_file = args[0]
    trained_model = options.trained_model

    parallel_prediction_main(para_file,trained_model)



if __name__ == '__main__':
    usage = "usage: %prog [options] para_file"
    parser = OptionParser(usage=usage, version="1.0 2021-04-07")
    parser.description = 'Introduction: run prediction using YOLOv4 '

    parser.add_option("-m", "--trained_model",
                      action="store", dest="trained_model",
                      help="the trained model for prediction")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)