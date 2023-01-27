#!/usr/bin/env python
# Filename: predict_yolov8.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 26 January, 2023
"""

import os, sys
import time
from datetime import datetime
from optparse import OptionParser

def main(options, args):

    para_file = args[0]
    trained_model = options.trained_model

    parallel_prediction_main(para_file,trained_model)



if __name__ == '__main__':
    usage = "usage: %prog [options] para_file"
    parser = OptionParser(usage=usage, version="1.0 2023-01-26")
    parser.description = 'Introduction: run prediction using YOLOv8 '

    parser.add_option("-m", "--trained_model",
                      action="store", dest="trained_model",
                      help="the trained model for prediction")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)