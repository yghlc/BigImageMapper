#!/usr/bin/env python
# Filename: parallel_predict_rts.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 24 September, 2019
"""

import os, sys

# get GPU information on the machine

# get


os.environ["CUDA_VISIBLE_DEVICES"] = "5"

os.system('echo $CUDA_VISIBLE_DEVICES')