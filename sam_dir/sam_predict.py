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

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters
import basic_src.io_function as io_function
import basic_src.basic as basic
import datasets.split_image as split_image
import datasets.raster_io as raster_io

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
    return patch_data

def save_masks_to_disk(accumulate_count, patch_boundary, masks,ref_raster, save_path,scores=None, b_prompt=False):
    # patch boundary: (xoff,yoff ,xsize, ysize)
    # Set output image data type based on the number of objects
    if len(masks) < 0:
        print('Warning, no masks')
        return False

    if b_prompt:
        dtype = np.uint8
        seg_map = None
        raise ('Something to do')
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

    raster_io.save_numpy_array_to_rasterfile(seg_map,save_path,ref_raster,compress='lzw', tiled='yes', bigtiff='if_safer',
                                             boundary=patch_boundary,verbose=False)

def segment_rs_image_sam(image_path, save_dir, model, model_type, patch_w, patch_h, overlay_x, overlay_y,
                        batch_size=1, prompts=None):

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    sam = sam_model_registry[model_type](checkpoint=model)
    if torch.cuda.is_available():
        sam.to(device='cuda')
    else:
        sam.to(device='cpu')

    if prompts is None:
        if overlay_x > 0 or overlay_y >0:
            raise ValueError('For everything mode, overlay_x and overlay_y should be zero')
        # segment everything
        mask_generator = SamAutomaticMaskGenerator(sam, output_mode="coco_rle")  # "binary_mask" take a lot of memory
    else:
        # only segment targets
        mask_generator = SamPredictor(sam)

    height, width, band_num, date_type = raster_io.get_height_width_bandnum_dtype(image_path)
    # print('input image: height, width, band_num, date_type',height, width, band_num, date_type)

    # read the entire image
    entire_img_data, nodata = raster_io.read_raster_all_bands_np(image_path)
    entire_img_data = entire_img_data.transpose(1, 2, 0)  # to opencv format  # in HWC uint8 format
    # # RGB to BGR: Matplotlib image to OpenCV https://www.scivision.dev/numpy-image-bgr-to-rgb/
    entire_img_data = entire_img_data[..., ::-1].copy()
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
        # img_data, nodata = raster_io.read_raster_all_bands_np(image_path, boundary=patches_sameSize[0])
        # img_data = img_data.transpose(1, 2, 0)
        # height, width, band_num = img_data.shape
        # if band_num not in [1, 3]:
        #     raise ValueError('only accept one band or three band images')

        # save file name
        file_name = "I%d_%d" % (0, p_idx)
        save_path = os.path.join(save_dir, file_name + '.tif')

        # yolov8 model can accept image with different size
        image = copy_one_patch_image_data(a_patch, entire_img_data)
        if prompts is None:
            masks = mask_generator.generate(image)
            save_masks_to_disk(total_seg_count,a_patch,masks, image_path,save_path)
            total_seg_count += len(masks)
        else:
            # generate masks based on input points
            mask_generator.set_image(image)
            masks, scores, logits = mask_generator.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            save_masks_to_disk(0,a_patch,masks,image_path, save_path,scores=scores, b_prompt=True)


        # if p_idx % 100 == 0:
        print('Processed %d patch, total: %d, this batch costs %f second' % (p_idx, patch_count, time.time() - t0))


def segment_remoteSensing_image(para_file, image_path, save_dir, network_ini, batch_size=1):
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

    model = parameters.get_file_path_parameters(network_ini,'checkpoint')
    model_type = parameters.get_string_parameters(network_ini,'model_type')

    # using the python API
    out = segment_rs_image_sam(image_path, save_dir, model, model_type,
                                         patch_w, patch_h, overlay_x, overlay_y, batch_size=batch_size)



def segment_one_image_sam(para_file, image_path, img_save_dir, inf_list_file, gpuid):

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

    segment_remoteSensing_image(para_file, image_path, img_save_dir, network_ini, batch_size=inf_batch_size)

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
                                  args=(para_file, inf_img_list[idx], img_save_dir, inf_list_file, gpuid))

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
