#!/usr/bin/env python
# Filename: parallel_predict_rts.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 24 September, 2019
"""

import os, sys
import os, sys
# HOME = os.path.expanduser('~')
# codes_dir2 = HOME +'/codes/PycharmProjects/DeeplabforRS'
# sys.path.insert(0, codes_dir2)
#
# import basic_src.basic as basic

import GPUtil

# get GPU information on the machine
# https://github.com/anderskm/gputil
deviceIDs = GPUtil.getAvailable(order = 'first', limit = 100, maxLoad = 0.5,
                                maxMemory = 0.5, includeNan=False, excludeID=[], excludeUUID=[])
print('available GPUs:',deviceIDs)



os.environ["CUDA_VISIBLE_DEVICES"] = str(deviceIDs[0])

os.system('echo $CUDA_VISIBLE_DEVICES')