#!/usr/bin/env python
# Filename: bim_utils.py 
"""
introduction: some common functions for the BigImageMapper

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 21 August, 2025
"""

import os,sys
import GPUtil

import time

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, code_dir)
import basic_src.basic as basic
import basic_src.io_function as io_function

def get_wait_available_GPU(machine_name, check_every_sec=5):
    # get available GPUs  # https://github.com/anderskm/gputil
    # memory: orders the available GPU device ids by ascending memory usage

    CUDA_VISIBLE_DEVICES = []
    if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
        CUDA_VISIBLE_DEVICES = [int(item.strip()) for item in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]

    while True:
        deviceIDs = GPUtil.getAvailable(order='memory', limit=100, maxLoad=0.5,
                                        maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])

        # only use the one in CUDA_VISIBLE_DEVICES
        if len(CUDA_VISIBLE_DEVICES) > 0:
            deviceIDs = [item for item in deviceIDs if item in CUDA_VISIBLE_DEVICES]
            print('on ' + machine_name + ', available GPUs:' + str(deviceIDs) +
                                   ', among visible ones:' + str(CUDA_VISIBLE_DEVICES))
        else:
            print('on ' + machine_name + ', available GPUs:' + str(deviceIDs))

        if len(deviceIDs) < 1:
            time.sleep(check_every_sec)  # wait some seconds, then check the available GPUs again
            continue

        return deviceIDs


def extract_sub_images(train_grids_shp,image_dir, buffersize,image_or_pattern,extract_img_dir,dstnodata,process_num,rectangle_ext,b_keep_org_file_name):

    get_subImage_script = os.path.join(code_dir, 'datasets', 'get_subImages.py')

    command_string = get_subImage_script + ' -b ' + str(buffersize) + ' -e ' + image_or_pattern + \
                     ' -o ' + extract_img_dir + ' -n ' + str(dstnodata) + ' -p ' + str(process_num) \
                      + ' --no_label_image '
    if b_keep_org_file_name:
        command_string += ' --b_keep_grid_name '
    if rectangle_ext:
        command_string += ' --rectangle '
    command_string += train_grids_shp + ' ' + image_dir
    basic.os_system_exit_code(command_string)

def rename_sub_images():
    pass



def main():
    pass


if __name__ == '__main__':
    main()
