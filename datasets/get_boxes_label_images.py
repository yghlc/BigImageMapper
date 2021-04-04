#!/usr/bin/env python
# Filename: get_boxes_label_images.py 
"""
introduction: for a given label image, get the box and the class_id

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 04 April, 2021
"""
import sys,os
from optparse import OptionParser

import cv2
import numpy as np

def get_boxes_from_label_image(label_path,nodata=None):
    '''
    get object boxes and ids from label images
    :param label_path:
    :param nodata:
    :return: 2d list for multiple boxes, [ class_id,  minX, minY, maxX, maxY ]
    '''

    # boxes [ class_id,  minX, minY, maxX, maxY ]

    # Load image
    label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)  # skimage.io.imread(label_path)
    # The label is a one band
    if label.ndim != 2:  # one band images only have a shape of (height, width), instead of (height, width, nband)
        raise ValueError('The band count of label must be 1')

    # set nodata pixels to background, that is 0
    if nodata is not None:
        label[label == nodata] = 0

    # height, width = label.shape
    unique_ids, counts = np.unique(label, return_counts=True)

    # if no objects on this images
    if max(unique_ids) == 0:
        # Call super class to return an empty mask
        return []           # no object
    else:

        contours, hierarchy = cv2.findContours(label, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # print contour for checking
        # for idx, con in enumerate(contours):
        #     print(idx, con.shape) # num, 1, 2
        #     num, _,_ = con.shape
        #     # print(idx, con)
        #     for ii in range(num):
        #         print(ii,con[ii,0,:],  label[con[ii,0,1], con[ii,0,0]]) # index, x,y, pixel value
        #         # test showing that, all pixels of contours are inside the objects.
        #     # break

        objects = []
        # find  [ class_id,  minX, minY, maxX, maxY ]
        for idx, con in enumerate(contours):
            num, _, _ = con.shape
            class_id = label[con[0,0,1], con[0,0,0]]
            minX =  con[0,0,0]
            maxX =  con[0,0,0]
            minY =  con[0,0,1]
            maxY =  con[0,0,1]
            for ii in range(1,num):
                if con[ii,0,0] < minX: minX = con[ii,0,0]
                if con[ii,0,0] > maxX: maxX = con[ii,0,0]
                if con[ii,0,1] < minY: minY = con[ii,0,1]
                if con[ii,0,1] > maxY: maxY = con[ii,0,1]
            objects.append([class_id, minX, minY, maxX, maxY])
            # print([class_id, minX, minY, maxX, maxY])


        # showing results for checking
        # label = label*100
        # cv2.imshow('label Image', label)
        # cv2.waitKey(0)
        # print("Number of Contours found = " + str(len(contours)))
        #
        # # Draw all contours
        # # -1 signifies drawing all contours
        # print( (objects[0][1], objects[0][2]), (objects[0][3], objects[0][4]) )
        # cv2.rectangle(label, (objects[0][1], objects[0][2]), (objects[0][3], objects[0][4]), (255), 1)    # draw a boxes
        # cv2.drawContours(label, contours, 0, (255), 1) # image, contours, contouridx (-1 for all), color, thickness
        #
        # cv2.imshow('Contours', label)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return objects


def test_get_boxes_from_label_image():
    dir = os.path.expanduser('~/Data/Arctic/canada_arctic/autoMapping/multiArea_yolov4_1/split_labels')
    label_path = os.path.join(dir, 'Banks_Island_mosaic_8bit_rgb_261_class_1_p_0.tif')
    get_boxes_from_label_image(label_path)


def main(options, args):
    label_path = args[0]
    get_boxes_from_label_image(label_path,nodata=options.nodata)


if __name__ == '__main__':
    usage = "usage: %prog [options] label_image "
    parser = OptionParser(usage=usage, version="1.0 2021-4-4")
    parser.description = 'Introduction: get boxes and class id from label images for objection detection.'

    parser.add_option("-n", "--nodata", type=int,
                      action="store", dest="nodata",
                      help="the nodata in label images")

    (options, args) = parser.parse_args()
    # print(options.no_label_image)
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)
