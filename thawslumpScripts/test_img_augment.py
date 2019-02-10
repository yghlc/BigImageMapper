#!/usr/bin/env python
# Filename: test_img_augment 
"""
introduction: perform the test of using different combination of data augmentation

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 09 February, 2019
"""

# use different combinations (not permutations) of image augmentation:
# spaces are not allow in img_aug_str
# only consider five of them: flip, blur, crop, scale, rotate


# output img_aug_str.txt, will be used in "test_img_augment.sh"

import os,sys

import subprocess

HOME = os.path.expanduser('~')
# path of DeeplabforRS
codes_dir2 = HOME +'/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)

import parameters



# get data augmentation already test
result_backup='result_backup'
# result_backup='/Users/huanglingcao/Dropbox/a_canada_sync'
img_aug_already_exist=[]

# get the path of all the porosity profile
file_pattern = os.path.join(result_backup, '*imgAug*.ini')
proc = subprocess.Popen('ls ' + file_pattern, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
profiles, err = proc.communicate()
ini_files = profiles.split()
for ini_fn in ini_files:
    img_aug_str = parameters.get_string_parameters(ini_fn,'data_augmentation')
    img_aug_already_exist.append(img_aug_str)


#  new data augmentation need to test

from itertools import combinations
test_id=0
with open('img_aug_str.txt','w') as f_obj:
    for count in range(1,6):
        comb = combinations(['flip', 'blur', 'crop', 'scale', 'rotate'], count)
        for idx, img_aug in enumerate(list(comb)):
            # spaces are not allow in img_aug_str
            img_aug_str=','.join(img_aug)

            test_id += 1

            # if this test already run, then skip it
            if img_aug_str in img_aug_already_exist:
                print(test_id, img_aug_str,'already exist')
                continue
            else:
                print(test_id, img_aug_str)
                f_obj.writelines(str(test_id)+' '+ img_aug_str+'\n')

