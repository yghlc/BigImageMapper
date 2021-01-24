#!/usr/bin/env python
# run "deeplab_train_test.py" for test, not "pytest deeplab_train_test.py"
# import unittest

import os, sys

code_dir = os.path.expanduser('~/codes/PycharmProjects/Landuse_DL')
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

    def test_init(self):
        # the global can only work for this file, not other files.
        # so it still complain cannot find parameters in "deeplab_train.py"

        global work_dir
        work_dir = os.path.expanduser('~/codes/PycharmProjects/Landuse_DL/working_dir')
        global para_file
        para_file = os.path.join(work_dir, 'main_para.ini')
        code_dir = os.path.expanduser('~/codes/PycharmProjects/Landuse_DL')

        # use "global to make these model can be use in other functions"
        sys.path.insert(0, code_dir)


    def test_get_train_val_sample_count(self):

        import parameters
        global parameters
        import basic_src.io_function as io_function
        global io_function
        import basic_src.basic as basic
        global basic

        print(deeplab_train.get_train_val_sample_count(work_dir, para_file))




if __name__ == '__main__':

    pass
