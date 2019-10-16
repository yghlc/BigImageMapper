#!/usr/bin/env python
# Filename: rm_unCompleted_task 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 16 October, 2019
"""

import os
dir = 'multi_inf_results'
for i in range(0,78):
    if os.path.isfile(os.path.join(dir,str(i)+'.txt')):
        if os.path.isfile(os.path.join(dir,str(i)+'.txt_done')):
            print('task %d is completed'%i)
        else:
            print('task %d is not completed, remove the folder' % i)
            folder = os.path.join(dir,str(i)+'I%d'%i)
            os.system('rm -rf '+ folder)

