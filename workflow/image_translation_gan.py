#!/usr/bin/env python
# Filename: image_translation_gan.py 
"""
introduction: using GAN (Generative Adversarial Networks) to convert images from one domain to another domain

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 07 October, 2021
"""

import io
import os,sys
import time
from datetime import datetime
from optparse import OptionParser

import GPUtil
from multiprocessing import Process

code_dir = os.path.join(os.path.dirname(sys.argv[0]), '..')
sys.path.insert(0, code_dir)
import parameters
import basic_src.basic as basic
import basic_src.io_function as io_function


def CUT_gan_is_ready_to_train(working_folder):
    if os.path.isfile(os.path.join(working_folder,'ready_to_train.txt')):
        return True
    return False


def train_CUT_gan(python_path, train_script,gan_para_file,gpu_ids):

    

    command_string = python_path + ' '  +  train_script \
                + ' os.chdir(current_dir)='+para_file \
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

    
    pass

def generate_image_CUT(python_path, generate_script):

    pass

def image_translate_train_generate_one_domain(gan_working_dir, gan_para_file, area_ini, gpu_ids, domainB_imgList):

    current_dir = os.getcwd()

    # get existing sub-images
    sub_img_label_txt = os.path.join(current_dir,'sub_images_labels_list.txt')
    if os.path.isfile(sub_img_label_txt) is False:
        raise IOError('%s not in the current folder, please get subImages first' % sub_img_label_txt)

    
    os.chdir(gan_working_dir)
    # prepare image list of domain A
    # what if the size of some images are not fit with CUT input?
    domain_A_images = []
    domain_A_labels = []
    with open(sub_img_label_txt) as txt_obj:
        line_list = [name.strip() for name in txt_obj.readlines()]
        for line in line_list:
            sub_image, sub_label = line.split(':')
            domain_A_images.append(sub_image)
            domain_A_labels.append(sub_label)
    io_function.save_list_to_txt(domain_A_images,'image_A_list.txt')

    # read target images, that will consider as target domains
    # what if there are too many images in domain B?
    io_function.save_list_to_txt(domainB_imgList,'image_B_list.txt')

    
    gan_python = parameters.get_file_path_parameters(gan_para_file,'python')
    cut_dir = parameters.get_directory(gan_para_file,'gan_script_dir')
    train_script = os.path.join(cut_dir, 'train.py')
    generate_script = os.path.join(cut_dir,'generate_image.py')
    # training of CUT 
    if train_CUT_gan(gan_python,train_script) is False:
        os.chdir(current_dir)
        return False

    # genenerate image using CUT





    # change working directory back
    os.chdir(current_dir)
    pass


def image_translate_train_generate_main(para_file, gpu_num):
    '''
     apply GAN to translate image from source domain to target domain

    existing sub-images (with sub-labels), these are image in source domain
    depend images for inference but no training data, each image for inference can be considered as on target domain

    '''
    print(datetime.now(), "image translation (train and generate) using GAN")

    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s'%(para_file, os.getcwd()))

    gan_para_file = parameters.get_string_parameters_None_if_absence(para_file, 'regions_n_setting_image_translation_ini')
    if gan_para_file is None:
        print('regions_n_setting_image_translation_ini is not set, skip image translation using GAN')
        return None

    machine_name = os.uname()[1]
    SECONDS = time.time()

    # get regions (equal to or subset of inference regions) need apply image translation
    multi_gan_regions = parameters.get_string_list_parameters(gan_para_file, 'regions_need_image_translation')
    gan_working_dir = parameters.get_string_parameters(gan_para_file, 'working_root')

    # loop each regions need image translation
    sub_tasks = []
    for area_idx, area_ini in enumerate(multi_gan_regions):

        area_name = parameters.get_string_parameters(area_ini, 'area_name')
        area_remark = parameters.get_string_parameters(area_ini, 'area_remark')
        area_time = parameters.get_string_parameters(area_ini, 'area_time')

        inf_image_dir = parameters.get_directory(area_ini, 'inf_image_dir')

        # it is ok consider a file name as pattern and pass it the following functions to get file list
        inf_image_or_pattern = parameters.get_string_parameters(area_ini, 'inf_image_or_pattern')

        inf_img_list = io_function.get_file_list_by_pattern(inf_image_dir, inf_image_or_pattern)
        img_count = len(inf_img_list)
        if img_count < 1:
            raise ValueError('No image for image translation, please check inf_image_dir and inf_image_or_pattern in %s' % area_ini)

        gan_project_save_dir = os.path.join(gan_working_dir, area_name + '_' + area_remark + '_' + area_time)

        if os.path.isdir(gan_project_save_dir):
            # TODO: check if results already exist
            pass
        else:
            io_function.mkdir(gan_project_save_dir)

        # parallel run image translation for this area
        CUDA_VISIBLE_DEVICES = []
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            CUDA_VISIBLE_DEVICES = [int(item.strip()) for item in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]

        # get an valid GPU
        gpuids = []
        while len(gpuids) < 1:
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
                print('No available GPUs, will check again in 60 seconds')
                time.sleep(60)  # wait one minute, then check the available GPUs again
                continue
            # set only the first available visible
            gpuids.append(deviceIDs[0])
            basic.outputlogMessage('%d:image translation for  %s on GPU %d of %s' % (area_idx, area_ini, gpuid, machine_name))

        # run image translation

        sub_process = Process(target=image_translate_train_generate_one_domain,
                              args=(gan_project_save_dir,gan_para_file, area_ini, gpuids,inf_img_list))

        sub_process.start()
        sub_tasks.append(sub_process)

        # wait until image translation has started or exceed 20 minutes
        time0 = time.time()
        elapsed_time = time.time() - time0
        while elapsed_time < 20 * 60:
            elapsed_time = time.time() - time0
            if CUT_gan_is_ready_to_train(gan_project_save_dir) is True or sub_process.is_alive() is False:
                break
            else:
                time.sleep(5)

        time.sleep(5)   # wait, allowing time for the GAN process to start.

        if sub_process.exitcode is not None and sub_process.exitcode != 0:
            sys.exit(1)

        basic.close_remove_completed_process(sub_tasks)

    # check all the tasks already finished
    while basic.b_all_process_finish(sub_tasks) is False:
        basic.outputlogMessage('wait all tasks to finish')
        time.sleep(60)
    basic.close_remove_completed_process(sub_tasks)


    duration= time.time() - SECONDS
    os.system('echo "$(date): time cost of tranlsate sub images to target domains: %.2f seconds">>time_cost.txt'%duration)


def main(options, args):
    para_file = sys.argv[1]
    gpu_num = int(sys.argv[2])

    image_translate_train_generate_main(para_file, gpu_num)



if __name__ == '__main__':
    usage = "usage: %prog [options] para_file gpu_num"
    parser = OptionParser(usage=usage, version="1.0 2021-10-07")
    parser.description = 'Introduction: translate images from source domain to target domain '

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)


