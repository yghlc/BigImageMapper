#!/usr/bin/env python
# Filename: image_translation_gan.py 
"""
introduction: using GAN (Generative Adversarial Networks) to convert images from one domain to another domain

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 07 October, 2021
"""

import os,sys
import time
from optparse import OptionParser

code_dir = os.path.join(os.path.dirname(sys.argv[0]), '..')
sys.path.insert(0, code_dir)
import parameters



def image_translate_tran_generate(para_file, gpu_num):
    '''
    # apply GAN to translate image from source domain to target domain

    existing sub-images (with sub-labels), these are image in source domain
    depend images for inference but no training data, each image for inference can be considered as on target domain

    '''
    print("image translation (train and generate) using GAN")

    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s'%(para_file, os.getcwd()))
    
    SECONDS = time.time()
    # get existing sub-images
    sub_img_label_txt = 'sub_images_labels_list.txt'
    if os.path.isfile(sub_img_label_txt) is False:
        raise IOError('%s not in the current folder, please get subImages first' % sub_img_label_txt)

    #

    # read target images, that will consider as target domains




    duration= time.time() - SECONDS
    os.system('echo "$(date): time cost of tranlsate sub images to target domains: %.2f seconds">>time_cost.txt'%duration)


def main(options, args):
    para_file = sys.argv[1]
    gpu_num = int(sys.argv[2])




if __name__ == '__main__':
    usage = "usage: %prog [options] para_file gpu_num"
    parser = OptionParser(usage=usage, version="1.0 2021-10-07")
    parser.description = 'Introduction: translate images from source domain to target domain '

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)


