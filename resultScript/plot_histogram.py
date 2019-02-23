#!/usr/bin/env python
# Filename: plot_histogram.py 
"""
introduction: plot histogram

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 23 February, 2019
"""

import os, sys
HOME = os.path.expanduser('~')
# path of DeeplabforRS
codes_dir2 =  HOME +'/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, codes_dir2)

import plot_results
import basic_src.io_function as io_function

out_dir=HOME+'/Data/Qinghai-Tibet/beiluhe/result/result_paper_mapping_RTS_dl_beiluhe'

# plot histogram of IOU values.
result_NOimgAug = HOME + '/Data/Qinghai-Tibet/beiluhe/result/result_paper_mapping_RTS_dl_beiluhe/' \
                         'BLH_basin_deeplabV3+_1_exp9_iter30000_post_1.shp'
result_imgAug16 = HOME + '/Data/Qinghai-Tibet/beiluhe/result/result_paper_mapping_RTS_dl_beiluhe/' \
                         'img_aug_test_results/BLH_basin_deeplabV3+_1_exp9_iter30000_post_imgAug16.shp'


# plot histogram of IOU values.
plot_results.draw_one_attribute_histogram(result_imgAug16, "IoU", "IoU (0-1)", "IoU_imgAug16.jpg")  # ,hatch='-'
plot_results.draw_one_attribute_histogram(result_NOimgAug, "IoU", "IoU (0-1)", "IoU_NOimgAug.jpg")

io_function.copy_file_to_dst('processLog.txt',out_dir+'/bins_iou.txt',overwrite=True)
io_function.copy_file_to_dst('IoU_imgAug16.jpg',out_dir+'/IoU_imgAug16.jpg',overwrite=True)
io_function.copy_file_to_dst('IoU_NOimgAug.jpg',out_dir+'/IoU_NOimgAug.jpg',overwrite=True)

