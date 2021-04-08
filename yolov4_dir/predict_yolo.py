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


def is_file_exist_in_folder(folder):
    # only check the first ten files
    # update on July 21, 2020. For some case, the first 10 may not exist (ignore if they are black)
    # so, if we find any file exist from 0 to 100000, then return True
    for i in range(100000):
        if os.path.isfile(os.path.join(folder, 'I0_%d.txt' % i)):
            return True
    return False
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

    split_image.split_image(image_path, save_dir, patch_w, patch_h, overlay_x, overlay_y, out_format, pre_name=None)
    # get list
    patch_list = io_function.get_file_list_by_ext(split_format,save_dir,bsub_folder=False)
    if len(patch_list) < 1:
        return None
    list_txt_path = save_dir + '_list.txt'
    io_function.save_list_to_txt(list_txt_path,patch_list)
    return list_txt_path


def predict_remoteSensing_image(para_file, image_path, save_dir,model, config_file, yolo_data, batch_siz=1, b_python_api=True):
    '''
    run prediction of a remote sensing using yolov4
    :param image_path:
    :param model:
    :param config_file:
    :param yolo_data:
    :param batch_siz:
    :param b_python_api: if true, use the python API of yolo
    :return:
    '''

    patch_w = parameters.get_digit_parameters(para_file, "inf_patch_width", 'int')
    patch_h = parameters.get_digit_parameters(para_file, "inf_patch_height", 'int')
    overlay_x = parameters.get_digit_parameters(para_file, "inf_pixel_overlay_x", 'int')
    overlay_y = parameters.get_digit_parameters(para_file, "inf_pixel_overlay_y", 'int')

    if b_python_api:
        # using the python API
        pass
    else:
        # divide image the many patches, then run prediction.
        patch_list_txt = split_an_image(para_file,image_path,save_dir,patch_w,patch_h,overlay_x,overlay_y)
        result_json = save_dir + '_result.json'
        commond_str = 'darknet detector test ' + yolo_data + ' ' + config_file + ' ' + model + ' -dont_show '
        commond_str += ' -ext_output -out ' + result_json + ' < ' + patch_list_txt
        print(commond_str)
        res = os.system(commond_str)
        if res !=0:
            sys.exit(1)


def predict_one_image_yolo(para_file, image_path, img_save_dir, inf_list_file, gpuid,trained_model):

    config_file = 'yolov4_obj.cfg'
    yolo_data = os.path.join('data','obj.data')
    b_python_api = False

    predict_remoteSensing_image(para_file,image_path, img_save_dir,trained_model, config_file, yolo_data, batch_siz=1, b_python_api=b_python_api)
    pass


def parallel_prediction_main(para_file, trained_model):

    print("YOLO prediction using the trained model (run parallel if use multiple GPUs)")
    machine_name = os.uname()[1]
    start_time = datetime.now()

    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    expr_name = parameters.get_string_parameters(para_file, 'expr_name')
    if trained_model is None:
        trained_model = os.path.join(expr_name, 'yolov4_obj_best.weights')

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
                deviceIDs = GPUtil.getAvailable(order='first', limit=100, maxLoad=0.5,
                                                maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
                # only use the one in CUDA_VISIBLE_DEVICES
                if len(CUDA_VISIBLE_DEVICES) > 0:
                    deviceIDs = [item for item in deviceIDs if item in CUDA_VISIBLE_DEVICES]
                    basic.outputlogMessage('on ' + machine_name + ', available GPUs:' + str(deviceIDs) +
                                           ', among visible ones:' + str(CUDA_VISIBLE_DEVICES))
                else:
                    basic.outputlogMessage('on ' + machine_name + ', available GPUs:' + str(deviceIDs))

                if len(deviceIDs) < 1:
                    time.sleep(60)  # wait one minute, then check the available GPUs again
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
                    time.sleep(5)

            idx += 1

            # wait until predicted image patches exist or exceed 20 minutes
            time0 = time.time()
            elapsed_time = time.time() - time0
            while elapsed_time < 20 * 60:
                elapsed_time = time.time() - time0
                file_exist = is_file_exist_in_folder(img_save_dir)
                if file_exist is True or sub_process.is_alive() is False:
                    break
                else:
                    time.sleep(5)

            if sub_process.exitcode is not None and sub_process.exitcode != 0:
                sys.exit(1)

            # if 'chpc' in machine_name:
            #     time.sleep(60)  # wait 60 second on ITSC services
            # else:
            #     time.sleep(10)

    # check all the tasks already finished
    while b_all_task_finish(sub_tasks) is False:
        basic.outputlogMessage('wait all tasks to finish')
        time.sleep(60)

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