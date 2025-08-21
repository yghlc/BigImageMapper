#!/usr/bin/env python
# Filename: bim_utils.py 
"""
introduction: some common funciton for the BigImageMapper

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 21 August, 2025
"""

import os
import GPUtil

import time

def get_wait_available_GPU(machine_name, check_every_sec=5):
    # get available GPUs  # https://github.com/anderskm/gputil
    # memory: orders the available GPU device ids by ascending memory usage
    while True:
        deviceIDs = GPUtil.getAvailable(order='memory', limit=100, maxLoad=0.5,
                                        maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])

        CUDA_VISIBLE_DEVICES = []
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            CUDA_VISIBLE_DEVICES = [int(item.strip()) for item in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]

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


def main():
    pass


if __name__ == '__main__':
    main()
