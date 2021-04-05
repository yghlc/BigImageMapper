#!/usr/bin/env python
# Filename: yolt_func 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 04 April, 2021
"""

# https://github.com/CosmiQ/simrdwn/blob/master/simrdwn/data_prep/yolt_data_prep_funcs.py
def convert(size, box):
    '''Input = image size: (w,h), box: [x0, x1, y0, y1]'''
    dw = 1./size[0]
    dh = 1./size[1]
    xmid = (box[0] + box[1])/2.0
    ymid = (box[2] + box[3])/2.0
    w0 = box[1] - box[0]
    h0 = box[3] - box[2]
    x = xmid*dw
    y = ymid*dh
    w = w0*dw
    h = h0*dh
    return (x, y, w, h)

# https://github.com/CosmiQ/simrdwn/blob/master/simrdwn/data_prep/yolt_data_prep_funcs.py
def convert_reverse(size, box):
    '''Back out pixel coords from yolo format
    input = image_size (w,h),
        box = [x,y,w,h]'''
    x, y, w, h = box
    dw = 1./size[0]
    dh = 1./size[1]

    w0 = w/dw
    h0 = h/dh
    xmid = x/dw
    ymid = y/dh

    x0, x1 = xmid - w0/2., xmid + w0/2.
    y0, y1 = ymid - h0/2., ymid + h0/2.

    return [x0, x1, y0, y1]
