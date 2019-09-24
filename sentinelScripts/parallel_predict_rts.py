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
HOME = os.path.expanduser('~')
codes_dir2 = HOME +'/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)

import basic_src.basic as basic

# get GPU information on the machine

gpu_str = basic  os.system('lspci | grep -i nvidia')
for line_str in gpu_str:
    if 'VGA compatible controller: NVIDIA' in line_str:


# get


os.environ["CUDA_VISIBLE_DEVICES"] = "5"

os.system('echo $CUDA_VISIBLE_DEVICES')