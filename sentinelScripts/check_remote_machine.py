#!/usr/bin/env python
# Filename: check_remote_machine 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 14 October, 2019
"""

import os, time

from copyTolocal_postPro import remote_workdir
from copyTolocal_postPro import get_remote_file_list
from copyTolocal_postPro import outdir
from copyTolocal_postPro import outputlogMessage

run_folder=os.path.join(remote_workdir, 'QTP_deeplabV3+_3')

while True:
    re_file_list = get_remote_file_list(os.path.join(run_folder, outdir, '*.txt_done'))

    if re_file_list is False:
        outputlogMessage('No completed prediction sub-images, wait 10 seconds ')
        time.sleep(10)  # wait one minute
        continue
    else:
        print('have completed prediction sub-images ')
        break