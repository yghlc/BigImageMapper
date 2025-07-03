#!/usr/bin/env python
# Filename: cluster_analysis.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 03 July, 2025
"""

import os,sys
from optparse import OptionParser

def load_clip_model(model_type, trained_model=None):
    pass

def get_img_features(image_list, model):
    """obtain the feature (vector) of images in laten space"""
    pass


def main(options, args):
    pass


if __name__ == '__main__':
    usage = "usage: %prog [options]  image_folder or image_list.txt "
    parser = OptionParser(usage=usage, version="1.0 2025-07-3")
    parser.description = 'Introduction: cluster analysis of small images using AI foundation models '

    parser.add_option("-s", "--trained_model",
                      action="store", dest="trained_model",
                      help="the trained model")

    # parser.add_option("-s", "--trained_clip_model",
    #                   action="store", dest="trained_clip_model",
    #                   help="the trained CLIP model")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)