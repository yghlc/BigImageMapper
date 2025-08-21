#!/usr/bin/env python
# Filename: predict_yolov8.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 26 January, 2023
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

import json
import  bim_utils
from multiprocessing import Process

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

def read_one_patch_image_data(patch, image_path):
    img_data, nodata = raster_io.read_raster_all_bands_np(image_path, boundary=patch)
    img_data = img_data.transpose(1, 2, 0)
    # # RGB to BGR: Matplotlib image to OpenCV https://www.scivision.dev/numpy-image-bgr-to-rgb/
    img_data = img_data[..., ::-1].copy()
    return img_data

def save_one_patch_yolov8_detection_json(patch_idx, patch, detections, class_names, save_dir, b_percent=False):
    # patch (xoff,yoff ,xsize, ysize)
    save_res_json = os.path.join(save_dir, '%d.json' % patch_idx)
    objects = []
    bbox_list = []
    boxes = detections.boxes.cpu().numpy()
    # xyxy:  box with xyxy format, (N, 4)
    for cls, confidence, bbox in zip( boxes.cls, boxes.conf, boxes.xyxy,):
        # print('cls, confidence, bbox',cls, confidence, bbox)
        # make sure # xmin >=0, ymin >=0, xmax<=xsize, ymax <= yszie
        bbox = [max(bbox[0], 0), max(bbox[1], 0), min(bbox[2], patch[2]), min(bbox[3], patch[3])]
        # remove many duplicate boxes if exists
        if bbox in bbox_list:
            # print('remove duplicated')
            continue
        else:
            bbox_list.append(bbox)

        # sometime, remove  very thin box
        if bbox[0] == bbox[2] or bbox[1] == bbox[3]:
            # print('xmin == xmax or ymin=ymax')
            continue

        bbox = [bbox[0] + patch[0], bbox[1] + patch[1], bbox[2] + patch[0],
                bbox[3] + patch[1]]  # to entire image coordinate

        if b_percent:
            confidence = round(confidence * 100, 2)
        object = {'class_id': int(cls),
                  'name': class_names[int(cls)],
                  'bbox': bbox,
                  'confidence': float(confidence)}
        objects.append(object)

    return objects
    # json_data = json.dumps(objects, indent=2)
    # with open(save_res_json, "w") as f_obj:
    #     f_obj.write(json_data)


def predict_rs_image_yolo8(image_path, save_dir, model, ultralytics_dir,class_names,
                           patch_w, patch_h, overlay_x, overlay_y, batch_size=1):
    if ultralytics_dir is not None:
        sys.path.insert(0, ultralytics_dir)
    from ultralytics import YOLO

    height, width, band_num, date_type = raster_io.get_height_width_bandnum_dtype(image_path)
    # print('input image: height, width, band_num, date_type',height, width, band_num, date_type)

    # divide the image the many small patches, then calcuate one by one, solving memory issues.
    image_patches = split_image.sliding_window(width, height, patch_w, patch_h, adj_overlay_x=overlay_x,
                                               adj_overlay_y=overlay_y)
    patch_count = len(image_patches)

    if os.path.isdir(save_dir) is False:
        io_function.mkdir(save_dir)

    batch_patches = [image_patches[i * batch_size: (i + 1) * batch_size] for i in
                     range((len(image_patches) + batch_size - 1) // batch_size)]

    model = YOLO(model)

    b_use_memory = True
    if height* width > 50000*50000:  # 10000*10000 is a threshold, can be changed
        b_use_memory = False

    # read the entire image
    if b_use_memory:
        entire_img_data, nodata = raster_io.read_raster_all_bands_np(image_path)
        entire_img_data = entire_img_data.transpose(1, 2, 0)  # to opencv format
        # # RGB to BGR: Matplotlib image to OpenCV https://www.scivision.dev/numpy-image-bgr-to-rgb/
        entire_img_data = entire_img_data[..., ::-1].copy()
        entire_height, entire_width, band_num = entire_img_data.shape
        print("entire_height, entire_width, band_num", entire_height, entire_width, band_num)
    if band_num not in [1, 3]:
        raise ValueError('only accept one band or three band images')

    patch_idx = 0
    all_objects = []
    with open(os.path.join(save_dir,'started.txt'),'w') as f_obj:
        f_obj.writelines(str(datetime.now()) + ': The process has started')
    for b_idx, a_batch_patch in enumerate(batch_patches):
        t0 = time.time()

        # yolov8 model can accept image with different size
        if b_use_memory:
            images = [copy_one_patch_image_data(patch, entire_img_data) for patch in a_batch_patch]
        else:
            images = [read_one_patch_image_data(patch, image_path) for patch in a_batch_patch]

        det_results = model(images, stream=True)  # generator of Results objects

        # save results
        objects = [save_one_patch_yolov8_detection_json(patch_idx + idx, patch, det_res,class_names, save_dir)
         for idx, (patch, det_res) in enumerate(zip(a_batch_patch,det_results))]
        [ all_objects.extend(item) for item in objects if len(item) > 0]   # ignore empty results


        if b_idx % 100 == 0:
            print('Processed %d patch, total: %d, this batch costs %f second' % (patch_idx + batch_size, patch_count, time.time() - t0))

        patch_idx += len(a_batch_patch)
    print('Have obtained results of all patches')
    return all_objects

def merge_patch_json_files_to_one(res_json_files, save_path):
    all_objects = []
    for idx, f_json in enumerate(res_json_files):
        objects = io_function.read_dict_from_txt_json(f_json)
        if len(objects) < 1:
            continue
        all_objects.extend(objects)
    json_data = json.dumps(all_objects, indent=2)
    with open(save_path, "w") as f_obj:
        f_obj.write(json_data)


def predict_remoteSensing_image(para_file, image_path, save_dir, model, network_ini, batch_size=1):
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
    object_names = parameters.get_string_list_parameters(para_file, 'object_names')

    ultralytics_dir = parameters.get_file_path_parameters_None_if_absence(network_ini,'ultralytics_dir')

    # using the python API
    all_objects = predict_rs_image_yolo8(image_path, save_dir, model, ultralytics_dir,object_names,
                                     patch_w, patch_h, overlay_x, overlay_y, batch_size=batch_size)

    # for each patch has a json file, may end up with a lot of json files, affect I/O
    # try to merge them to one json file.
    # res_json_files = io_function.get_file_list_by_ext('.json', save_dir, bsub_folder=False)
    merge_josn_path = os.path.join(save_dir,'all_patches.json')
    # merge_patch_json_files_to_one(res_json_files,merge_josn_path)
    # for f_json in res_json_files:
    #     io_function.delete_file_or_dir(f_json)
    json_data = json.dumps(all_objects, indent=2)
    with open(merge_josn_path, "w") as f_obj:
        f_obj.write(json_data)



def predict_one_image_yolov8(para_file, image_path, img_save_dir, inf_list_file, gpuid,trained_model):

    config_file = parameters.get_string_parameters(para_file, 'network_setting_ini')  # 'yolov4_obj.cfg'
    inf_batch_size = parameters.get_digit_parameters(para_file,'inf_batch_size','int')

    done_indicator = '%s_done'%inf_list_file
    if os.path.isfile(done_indicator):
        basic.outputlogMessage('warning, %s exist, skip prediction'%done_indicator)
        return
    # use a specific GPU for prediction, only inference one image
    time0 = time.time()
    if gpuid is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)

    predict_remoteSensing_image(para_file,image_path, img_save_dir,trained_model, config_file, batch_size=inf_batch_size)

    duration = time.time() - time0
    os.system('echo "$(date): time cost of inference for image in %s: %.2f seconds">>"time_cost.txt"' % (inf_list_file, duration))
    # write a file to indicate that the prediction has done.
    os.system('echo %s > %s_done'%(inf_list_file,inf_list_file))
    return

def test_predict_one_image_yolov8():
    work_dir = os.path.expanduser('~/Data/tmp_data/yolov8_test/panArctic_yolov8_1/')
    para_file = 'main_para.ini'
    image_path = os.path.join(work_dir, 'subImages', 'ext00_hillshade_newest_HWLine_33_class_1.tif')
    img_save_dir = os.path.join(work_dir,'prediction','I0')
    inf_list_file = '0.txt'
    gpuid = None
    trained_model = os.path.join(work_dir,'exp1','weights','best.pt')
    predict_one_image_yolov8(para_file, image_path, img_save_dir, inf_list_file, gpuid, trained_model)

def parallel_prediction_main(para_file, trained_model):

    print("YOLO prediction using the trained model (run parallel if using multiple GPUs)")
    machine_name = os.uname()[1]
    start_time = datetime.now()

    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    expr_name = parameters.get_string_parameters(para_file, 'expr_name')
    network_ini = parameters.get_string_parameters(para_file,'network_setting_ini')
    if trained_model is None:
        trained_model = os.path.join(expr_name, 'weights', 'best.pt')

    outdir = parameters.get_directory(para_file, 'inf_output_dir')
    # remove previous results (let user remove this folder manually or in exe.sh folder)
    io_function.mkdir(outdir)

    # get name of inference areas
    multi_inf_regions = parameters.get_string_list_parameters(para_file, 'inference_regions')
    b_use_multiGPUs = parameters.get_bool_parameters(para_file, 'b_use_multiGPUs')
    maximum_prediction_jobs = parameters.get_digit_parameters(para_file,'maximum_prediction_jobs','int')

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
            raise ValueError('No image for inference, please check inf_image_dir (%s) and inf_image_or_pattern (%s) in %s'
                             % (inf_image_dir,inf_image_or_pattern,area_ini))

        area_save_dir = os.path.join(outdir, area_name + '_' + area_remark + '_' + area_time)
        io_function.mkdir(area_save_dir)

        # parallel inference images for this area
        idx = 0
        while idx < img_count:

            img_save_dir = os.path.join(area_save_dir, 'I%d' % idx)
            inf_list_file = os.path.join(area_save_dir, '%d.txt' % idx)

            done_indicator = '%s_done' % inf_list_file
            if os.path.isfile(done_indicator):
                basic.outputlogMessage('warning, %s exist, skip prediction' % done_indicator)
                idx += 1
                continue

            while basic.alive_process_count(sub_tasks) >= maximum_prediction_jobs:
                print(datetime.now(),'%d jobs are running simultaneously, wait 5 seconds'%basic.alive_process_count(sub_tasks))
                time.sleep(5)   # wait 5 seconds, then check the count of running jobs again

            if b_use_multiGPUs:
                deviceIDs = bim_utils.get_wait_available_GPU(machine_name, check_every_sec=5)

                # set only the first available visible
                gpuid = deviceIDs[0]
                basic.outputlogMessage('%d: predict image %s on GPU %d of %s' % (idx, inf_img_list[idx], gpuid, machine_name))
            else:
                gpuid = None
                basic.outputlogMessage('%d: predict image %s on %s' % (idx, inf_img_list[idx], machine_name))

            # run inference

            # if it already exist, then skip
            if os.path.isdir(img_save_dir) and is_file_exist_in_folder(img_save_dir):
                basic.outputlogMessage('folder of %dth image (%s) already exist, '
                                       'it has been predicted or is being predicted' % (idx, inf_img_list[idx]))
                idx += 1
                continue

            with open(inf_list_file, 'w') as inf_obj:
                inf_obj.writelines(inf_img_list[idx] + '\n')

            sub_process = Process(target=predict_one_image_yolov8,
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
    trained_model = options.trained_model

    # test_predict_one_image_yolov8()
    parallel_prediction_main(para_file,trained_model)



if __name__ == '__main__':
    usage = "usage: %prog [options] para_file"
    parser = OptionParser(usage=usage, version="1.0 2023-01-26")
    parser.description = 'Introduction: run prediction using YOLOv8 '

    parser.add_option("-m", "--trained_model",
                      action="store", dest="trained_model",
                      help="the trained model for prediction")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)