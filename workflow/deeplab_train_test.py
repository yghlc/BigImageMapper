#!/usr/bin/env python
# run "pytest deeplab_train_test.py " or "pytest " for test, add " -s for allowing print out"
# "pytest can automatically search *_test.py files "
# import unittest

import os, sys

# the path of Landuse_DL
# code_dir = os.path.expanduser('~/codes/PycharmProjects/Landuse_DL')
code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')   # get path of deeplab_train_test.py
print(code_dir)
sys.path.insert(0, code_dir)


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


    # def test_get_train_val_sample_count(self):
    #
    #     print(deeplab_train.get_train_val_sample_count(work_dir, para_file))
    #
    # def test_get_trained_iteration(self):
    #     train_log_dir = os.path.join(work_dir, 'exp1', 'train')
    #     iter = deeplab_train.get_trained_iteration(train_log_dir)
    #     print('iteration number in the folder', iter)

    def test_get_miou_spep_list(self):
        train_log_dir = os.path.join(work_dir, 'exp1', 'eval')
        dict = deeplab_train.get_miou_list_class_all(train_log_dir,2)
        print(dict)

    def test_get_loss_list(self):
        train_log_dir = os.path.join(work_dir, 'exp1', 'train')
        dict = deeplab_train.get_loss_learning_rate_list(train_log_dir)
        print(dict)



if __name__ == '__main__':

    pass
