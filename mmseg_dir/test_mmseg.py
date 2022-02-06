#!/usr/bin/env python
# Filename: test_mmseg.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 17 January, 2022
"""

import os,sys
import os.path as osp

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)

import basic_src.io_function as io_function
import parameters
from mmseg_predict import predict_one_image_mmseg
from workflow.postProcess import inf_results_to_shapefile

# sys.setrecursionlimit(100)

def set_pythonpath(para_file):

    expr_name = parameters.get_string_parameters(para_file, 'expr_name')
    network_ini = parameters.get_string_parameters(para_file, 'network_setting_ini')
    mmseg_repo_dir = parameters.get_directory(network_ini, 'mmseg_repo_dir')
    mmseg_code_dir = osp.join(mmseg_repo_dir,'mmseg')

    if os.path.isdir(mmseg_code_dir) is False:
        raise ValueError('%s does not exist' % mmseg_code_dir)

    # set PYTHONPATH to use my modified version of mmseg
    if os.getenv('PYTHONPATH'):
        os.environ['PYTHONPATH'] = os.getenv('PYTHONPATH') + ':' + mmseg_code_dir
    else:
        os.environ['PYTHONPATH'] = mmseg_code_dir
    print('\nPYTHONPATH is: ',os.getenv('PYTHONPATH'))

def test_dataloader():
    # test, run in ~/Data/tmp_data/test_mmsegmentation/test_landuse_dl
    para_file = 'main_para.ini'
    set_pythonpath(para_file)

    # test rgb, using rgb in Willow River
    # img_idx = 0
    # image_path = os.path.expanduser('~/Data/Arctic/canada_arctic/Willow_River/Planet2020/20200818_mosaic_8bit_rgb.tif')
    # img_save_dir = os.path.join('predict_output','I%d' % img_idx)
    # io_function.mkdir(img_save_dir)
    # inf_list_file = os.path.join('predict_output','%d.txt'%img_idx)
    # gpuid = None
    # trained_model = 'exp18/latest.pth'
    # predict_one_image_mmseg(para_file, image_path, img_save_dir, inf_list_file, gpuid, trained_model)
    #
    # # curr_dir,img_idx, area_save_dir, test_id
    # curr_dir = os.getcwd()
    # inf_results_to_shapefile(curr_dir,img_idx,'predict_output','1')

    ############ test nirGB, using rgb in Willow River
    img_idx = 1
    image_path = os.path.expanduser('~/Data/Arctic/canada_arctic/Willow_River/Planet2020/20200818_mosaic_8bit_nirGB.tif')
    img_save_dir = os.path.join('predict_output','I%d' % img_idx)
    io_function.mkdir(img_save_dir)
    inf_list_file = os.path.join('predict_output','%d.txt'%img_idx)
    gpuid = None
    trained_model = 'exp18/latest.pth'
    predict_one_image_mmseg(para_file, image_path, img_save_dir, inf_list_file, gpuid, trained_model)

    # curr_dir,img_idx, area_save_dir, test_id
    curr_dir = os.getcwd()
    inf_results_to_shapefile(curr_dir,img_idx,'predict_output','1')



def main():
    test_dataloader()
    pass

if __name__ == '__main__':
    main()


