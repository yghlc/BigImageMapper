#!/usr/bin/env python
# Filename: test_scripts 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 11 April, 2021
"""
import os, sys
import cv2
import numpy as np

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)

import datasets.raster_io as raster_io

def test_read_image_cv2_rasterio():

    # run in ~/Data/Arctic/canada_arctic/autoMapping/multiArea_yolov4_1
    dir = os.path.expanduser('~/Data/Arctic/canada_arctic/autoMapping/multiArea_yolov4_1')
    print('\n')
    print('Run read_image_cv2_rasterio')

    img_path = os.path.join(dir,'debug_img', '20200818_mosaic_8bit_rgb_0_class_1_p_0.png')
    image_cv2 = cv2.imread(img_path)
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)  # BGR to RGB
    print('image_cv2 shape', image_cv2.shape)
    for band in range(3):
        data = image_cv2[:,:,band]
        print('b %d'%band, np.mean(data))

    image_rs, nodata = raster_io.read_raster_all_bands_np(img_path)
    image_rs = image_rs.transpose(1, 2, 0)
    print('image_rs shape', image_rs.shape)
    for band in range(3):
        data = image_rs[:,:,band]
        print('b %d'%band, np.mean(data))

    # both image_cv2 and image_rs have date type of 'numpy.ndarray', but when draw rectange on image_rs,
    # it complains not TypeError: Expected Ptr<cv::UMat> for argument 'img'
    print('type of image_cv2',type(image_cv2))
    print('type of image_rs',type(image_rs))

    image_cv2 = cv2.rectangle(image_cv2, (10, 10), (100, 100), (0,0,0), 1)
    # as suggested by https://stackoverflow.com/questions/57586449/why-cv2-rectangle-sometimes-return-np-ndarray-while-sometimes-cv2-umat
    # change to contiguous, then it passes
    image_rs = np.ascontiguousarray(image_rs)
    image_rs = cv2.rectangle(image_rs, (50, 50), (150, 150), (255,255,255), 1)

    cv2.imshow('image_cv2', image_cv2)
    cv2.imshow('image_rs', image_rs)
    if cv2.waitKey() & 0xFF == ord('q'):
        return
