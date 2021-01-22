#!/usr/bin/env python
# Filename: parallel_predict_rts.py
"""
introduction: parallel run inference (prediction) using multiple GPUs

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 24 September, 2019
"""

# note: it seems the codes cannot run on multi-nodes on ITSC services. So I have to submit jobs using different separately

import os, sys
import time

import GPUtil
import datetime
from multiprocessing import Process

def is_file_exist_in_folder(folder):
    # only check the first ten files
    # update on July 21, 2020. For some case, the first 10 may not exist (ignore if they are black)
    # so, if we find any file exist from 0 to 1000000, then return True
    for i in range(1000000):
        if os.path.isfile(os.path.join(folder, 'I0_%d.tif' % i)):
            return True
    return False
    # file_list = io_function.get_file_list_by_pattern(folder, '*.*')  # this may take time if a lot of file exist
    # if len(file_list) > 0:
    #     return True
    # else:
    #     return False


def predict_one_image_deeplab(deeplab_inf_script, para_file,network_ini, save_dir,inf_list_file,gpuid=None):

    # use a specific GPU for prediction, only inference one image
    time0 = time.time()
    if gpuid is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid

    # command_string = deeplab_predict_script + ' '+ para_file + ' ' + save_dir + ' ' + inf_list_file + ' ' + str(gpuid)
    # # status, result = basic.exec_command_string(command_string)  # this will wait command finished
    # # os.system(command_string + "&")  # don't know when it finished
    # os.system(command_string )      # this work

    WORK_DIR = os.getcwd()
    iteration_num = parameters.get_digit_parameters_None_if_absence(network_ini, 'export_iteration_num', 'int')
    expr_name = parameters.get_string_parameters(para_file, 'expr_name')
    EXP_FOLDER = expr_name
    EXPORT_DIR = os.path.join(WORK_DIR, EXP_FOLDER, 'export')
    EXPORT_PATH = os.path.join(EXPORT_DIR, 'frozen_inference_graph_%s.pb' % iteration_num)
    frozen_graph_path = EXPORT_PATH

    inf_batch_size = parameters.get_digit_parameters_None_if_absence(network_ini,'inf_batch_size','int')
    if inf_batch_size is None:
        raise ValueError('inf_batch_size not set in %s'%network_ini)

    command_string = 'python '  +  deeplab_inf_script \
                + ' --inf_para_file='+para_file \
                + ' --inf_list_file='+inf_list_file \
                + ' --inf_batch_size='+str(inf_batch_size) \
                + ' --inf_output_dir='+save_dir \
                + ' --frozen_graph_path='+frozen_graph_path
    # status, result = basic.exec_command_string(command_string)  # this will wait command finished
    # os.system(command_string + "&")  # don't know when it finished
    res = os.system(command_string)  # this work
    # print('command_string deeplab_inf_script: res',res)
    if res != 0:
        sys.exit(1)

    duration = time.time() - time0
    os.system('echo "$(date): time cost of inference for image in %s: %.2f seconds">>"time_cost.txt"' % (inf_list_file, duration))
    # write a file to indicate that the prediction has done.
    os.system('echo %s > %s_done'%(inf_list_file,inf_list_file))



def b_all_task_finish(all_tasks):
    for task in all_tasks:
        if task.is_alive():
            return False
    return True


if __name__ == '__main__':

    print("%s : export the frozen inference graph" % os.path.basename(sys.argv[0]))
    machine_name = os.uname()[1]
    start_time = datetime.datetime.now()

    para_file = sys.argv[1]
    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    code_dir = os.path.join(os.path.dirname(sys.argv[0]), '..')
    sys.path.insert(0, code_dir)
    import parameters

    deeplabRS = parameters.get_directory(para_file, 'deeplabRS_dir')
    sys.path.insert(0, deeplabRS)
    import basic_src.io_function as io_function
    import basic_src.basic as basic

    basic.setlogfile('parallel_predict_Log.txt')

    deeplab_inf_script = os.path.join(code_dir,'deeplabBased','deeplab_inference.py')
    network_setting_ini = parameters.get_string_parameters(para_file, 'network_setting_ini')

    outdir = parameters.get_directory(para_file,'inf_output_dir')

    # remove previous results (let user remove this folder manually or in exe.sh folder)
    io_function.mkdir(outdir)

    # get name of inference areas
    multi_inf_regions = parameters.get_string_list_parameters(para_file, 'inference_regions')

    max_parallel_inf_task = parameters.get_digit_parameters(para_file,'max_parallel_inf_task','int')

    b_use_multiGPUs = parameters.get_bool_parameters(para_file,'b_use_multiGPUs')

    # loop each inference regions
    sub_tasks = []
    for area_idx, area_ini in enumerate(multi_inf_regions):

        area_name = parameters.get_string_parameters_None_if_absence(area_ini,'area_name')
        area_remark = parameters.get_string_parameters_None_if_absence(area_ini,'area_remark')

        inf_image_dir = parameters.get_directory_None_if_absence(area_ini, 'inf_image_dir')
        if inf_image_dir is None:
            raise ValueError('%s not set in %s'%('inf_image_dir', area_ini))
        # it is ok consider a file name as pattern and pass it the following functions to get file list
        inf_image_or_pattern = parameters.get_string_parameters_None_if_absence(area_ini, 'inf_image_or_pattern')
        if inf_image_or_pattern is None:
            raise ValueError('%s not set in %s' % ('inf_image_or_pattern', area_ini))

        inf_img_list = io_function.get_file_list_by_pattern(inf_image_dir,inf_image_or_pattern)
        img_count = len(inf_img_list)
        if img_count < 1:
            raise ValueError('No image for inference, please check inf_image_dir and inf_image_or_pattern in %s'%area_ini)

        area_save_dir = os.path.join(outdir, area_name + '_' + area_remark)
        io_function.mkdir(area_save_dir)

        # parallel inference images for this area
        idx = 0
        while idx < img_count:

            if b_use_multiGPUs:
                # get available GPUs  # https://github.com/anderskm/gputil
                deviceIDs = GPUtil.getAvailable(order='first', limit=100, maxLoad=0.5,
                                                maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
                basic.outputlogMessage('on ' + machine_name + ', available GPUs:'+str(deviceIDs))
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
            img_save_dir = os.path.join(area_save_dir,'I%d'%idx)
            inf_list_file = os.path.join(area_save_dir,'%d.txt'%idx)

            # if it already exist, then skip
            if os.path.isdir(img_save_dir) and is_file_exist_in_folder(img_save_dir):
                basic.outputlogMessage('folder of %dth image (%s) already exist, '
                                       'it has been predicted or is being predicted'%(idx, inf_img_list[idx]))
                idx += 1
                continue

            with open(inf_list_file,'w') as inf_obj:
                inf_obj.writelines(inf_img_list[idx] + '\n')

            sub_process = Process(target=predict_one_image_deeplab,
                                  args=(deeplab_inf_script, para_file,network_setting_ini,img_save_dir,inf_list_file,gpuid))
            sub_process.start()
            sub_tasks.append(sub_process)

            idx += 1

            # wait until predicted image patches exist or exceed 20 minutes
            time0 = time.time()
            elapsed_time = time.time() - time0
            while elapsed_time < 20*60:
                elapsed_time = time.time() - time0
                file_exist = is_file_exist_in_folder(img_save_dir)
                if file_exist is True or sub_process.is_alive() is False:
                    break
                else:
                    time.sleep(5)

            # if 'chpc' in machine_name:
            #     time.sleep(60)  # wait 60 second on ITSC services
            # else:
            #     time.sleep(10)

    # check all the tasks already finished
    while b_all_task_finish(sub_tasks) is False:
        basic.outputlogMessage('wait all tasks to finish')
        time.sleep(60)


    end_time = datetime.datetime.now()

    diff_time = end_time - start_time
    out_str = "%s: time cost of parallel inference on %s: %d seconds"%(str(end_time),machine_name,diff_time.seconds)
    basic.outputlogMessage(out_str)
    with open ("time_cost.txt",'a') as t_obj:
        t_obj.writelines(out_str+'\n')

