#!/usr/bin/env python
# Filename: clip_train.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 24 January, 2024
"""

import os,sys
from optparse import OptionParser


def main(options, args):

    pass


if __name__ == '__main__':
    usage = "usage: %prog [options] para_file"
    parser = OptionParser(usage=usage, version="1.0 2024-01-24")
    parser.description = 'Introduction: fine-tune the clip model using custom data'

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)