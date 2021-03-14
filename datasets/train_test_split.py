#!/usr/bin/env python
# Filename: train_test_split 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 17 November, 2018
"""

import sys,os,subprocess
from optparse import OptionParser

from sklearn.model_selection import train_test_split

def train_test_split_main(input_file,train_per,Do_shuffle,train_sample_txt,val_sample_txt):

    with open(input_file,'r') as f_obj:
        dir = os.path.dirname(input_file)

        files_list = f_obj.readlines()

        train_list, val_list = train_test_split(files_list, train_size=train_per, shuffle=Do_shuffle)

        # save split list
        train_list_txt = os.path.join(dir,train_sample_txt)
        val_list_txt = os.path.join(dir, val_sample_txt)
        with open(train_list_txt, 'w') as t_obj:
            t_obj.writelines(train_list)
            print('saved training samples to %s'%train_list_txt)

        with open(val_list_txt, 'w') as v_obj:
            v_obj.writelines(val_list)
            print('saved validation samples to %s' % val_list_txt)

def main(options, args):

    input_file = args[0]
    train_per = options.train_per
    Do_shuffle = options.Do_shuffle

    print('split images in %s to train and test, with'%input_file)
    print('train percentage: %.4f and shuffle: %s'%(train_per,str(Do_shuffle)))

    train_sample_txt = options.train_list_txt
    val_sample_txt = options.val_list_txt

    train_test_split_main(input_file, train_per, Do_shuffle, train_sample_txt, val_sample_txt)




if __name__ == "__main__":
    usage = "usage: %prog [options] image_list "
    parser = OptionParser(usage=usage, version="1.0 2018-11-17")
    parser.description = 'Introduction: split images to subset of train and test '

    parser.add_option('-p','--train_per',
                      action='store',dest='train_per',type='float',default=0.8,
                      help="percentage of training data, a float value from 0 to 1")

    parser.add_option('-t','--train_list_txt',
                      action='store',dest='train_list_txt',default ='train_list.txt',
                      help="the txt file name for saving training samples")

    parser.add_option('-v','--val_list_txt',
                      action='store',dest='val_list_txt',default ='val_list.txt',
                      help="the txt file name for saving validation samples")

    parser.add_option('-s','--shuffle',
                      action='store_true',dest='Do_shuffle',default=False,
                      help="shuffle before splitting")


    (options, args) = parser.parse_args()
    if len(sys.argv) < 2 or len(args) < 1:
        parser.print_help()
        sys.exit(2)

    main(options, args)


