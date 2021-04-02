#!/usr/bin/env python
# Filename: yoltv4_common.py 
"""
introduction: some common variable for yoltv4

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 01 April, 2021
"""


import os

# path for yoltv4
yoltv4_dir = os.path.expanduser('~/codes/PycharmProjects/yghlc_yoltv4')
darknet_dir = os.path.join(yoltv4_dir,'darknet')
darknet_bin = os.path.join(darknet_dir,'darknet')

print(darknet_bin)


# path for