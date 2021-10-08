#!/usr/bin/env python
# Filename: image_translation_gan.py 
"""
introduction: using GAN (Generative Adversarial Networks) to convert images from one domain to another domain

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 07 October, 2021
"""

import os,sys
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(sys.argv[0]), '..')
sys.path.insert(0, code_dir)
import parameters


def main(options, args):
    pass

if __name__ == '__main__':
    usage = "usage: %prog [options] para_file gpu_num"
    parser = OptionParser(usage=usage, version="1.0 2021-10-07")
    parser.description = 'Introduction: translate images from source domain to target domain '

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)


