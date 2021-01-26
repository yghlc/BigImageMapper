#!/usr/bin/env python
# run "pytest deeplab_train_test.py " or "pytest " for test, add " -s for allowing print out"
# "pytest can automatically search *_test.py files "
# import unittest

import os, sys

# code_dir = os.path.expanduser('~/codes/PycharmProjects/Landuse_DL')
# sys.path.insert(0, code_dir)


work_dir = os.path.expanduser('~/codes/PycharmProjects/Landuse_DL/working_dir')
os.chdir(work_dir)
para_file = 'main_para.ini'

import workflow.deeplab_train as deeplab_train

class TestdeeplabTrainclass():
    #  cannot collect test class 'TestdeeplabTrainclass' because it has a __init__ constructor
    # def __init__(self):
    #     self.work_dir = None
    #     self.code_dir = None
    #     self.para_file = None


    def test_get_train_val_sample_count(self):

        print(deeplab_train.get_train_val_sample_count(work_dir, para_file))

    def test_get_trained_iteration(self):
        train_log_dir = os.path.join(work_dir, 'exp1', 'train')
        iter = deeplab_train.get_trained_iteration(train_log_dir)
        print('iteration number in the folder', iter)



if __name__ == '__main__':

    pass
