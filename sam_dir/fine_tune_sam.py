#!/usr/bin/env python
# Filename: fine_tune_sam.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 06 April, 2024
"""

import os, sys
import time
from datetime import datetime
from optparse import OptionParser

import pandas as pd

def fine_tune_sam_main(para_file):
    pass

def main(options, args):

    para_file = args[0]
    fine_tune_sam_main(para_file)

if __name__ == '__main__':
    usage = "usage: %prog [options] para_file"
    parser = OptionParser(usage=usage, version="1.0 2024-04-06")
    parser.description = 'Introduction: fine-tuning segment anything models '

    # parser.add_option("-m", "--trained_model",
    #                   action="store", dest="trained_model",
    #                   help="the trained model for prediction")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)
