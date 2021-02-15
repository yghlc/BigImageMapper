#!/usr/bin/env python
# run "pytest deeplab_train_test.py " or "pytest " for test, add " -s for allowing print out"
# "pytest can automatically search *_test.py files "
# import unittest

import os, sys
import time

from multiprocessing import Process

# the path of Landuse_DL
# code_dir = os.path.expanduser('~/codes/PycharmProjects/Landuse_DL')
code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')   # get path of deeplab_train_test.py
print(code_dir)
sys.path.insert(0, code_dir)
import parameters

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

    # def test_get_miou_spep_list(self):
    #     train_log_dir = os.path.join(work_dir, 'exp1', 'eval')
    #     dict = deeplab_train.get_miou_list_class_all(train_log_dir,2)
    #     print(dict)
    #
    # def test_get_loss_list(self):
    #     train_log_dir = os.path.join(work_dir, 'exp1', 'train')
    #     dict = deeplab_train.get_loss_learning_rate_list(train_log_dir)
    #     print(dict)

    # def test_evaluation_deeplab(self):
    #
    #     # run this test "pytest -s deeplab_train_test.py" in
    #     # ~/Data/Arctic/canada_arctic/autoMapping/multiArea_deeplabV3+_6 or other working folder (with trained model and data avaible)
    #
    #     para_file = 'main_para.ini'
    #
    #     if os.path.isfile(para_file) is False:
    #         raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))
    #
    #     network_setting_ini = parameters.get_string_parameters(para_file, 'network_setting_ini')
    #     tf_research_dir = parameters.get_directory_None_if_absence(network_setting_ini, 'tf_research_dir')
    #     print(tf_research_dir)
    #     if tf_research_dir is None:
    #         raise ValueError('tf_research_dir is not in %s' % para_file)
    #     if os.path.isdir(tf_research_dir) is False:
    #         raise ValueError('%s does not exist' % tf_research_dir)
    #
    #     if os.getenv('PYTHONPATH'):
    #         os.environ['PYTHONPATH'] = os.getenv('PYTHONPATH') + ':' + tf_research_dir + ':' + os.path.join(
    #             tf_research_dir,
    #             'slim')
    #     else:
    #         os.environ['PYTHONPATH'] = tf_research_dir + ':' + os.path.join(tf_research_dir, 'slim')
    #
    #     global tf1x_python
    #     tf1x_python = parameters.get_file_path_parameters(network_setting_ini, 'tf1x_python')
    #
    #     WORK_DIR = os.getcwd()
    #     expr_name = parameters.get_string_parameters(para_file, 'expr_name')
    #     deeplab_dir = os.path.join(tf_research_dir, 'deeplab')
    #
    #     # prepare training folder
    #     EXP_FOLDER = expr_name
    #     TRAIN_LOGDIR = os.path.join(WORK_DIR, EXP_FOLDER, 'train')
    #     EVAL_LOGDIR = os.path.join(WORK_DIR, EXP_FOLDER, 'eval')
    #
    #     dataset_dir = os.path.join(WORK_DIR, 'tfrecord')
    #
    #     inf_output_stride = parameters.get_digit_parameters_None_if_absence(network_setting_ini, 'inf_output_stride','int')
    #     inf_atrous_rates1 = parameters.get_digit_parameters_None_if_absence(network_setting_ini, 'inf_atrous_rates1','int')
    #     inf_atrous_rates2 = parameters.get_digit_parameters_None_if_absence(network_setting_ini, 'inf_atrous_rates2', 'int')
    #     inf_atrous_rates3 = parameters.get_digit_parameters_None_if_absence(network_setting_ini, 'inf_atrous_rates3','int')
    #
    #     # depth_multiplier default is 1.0.
    #     depth_multiplier = parameters.get_digit_parameters_None_if_absence(network_setting_ini, 'depth_multiplier','float')
    #
    #     decoder_output_stride = parameters.get_digit_parameters_None_if_absence(network_setting_ini,'decoder_output_stride', 'int')
    #     aspp_convs_filters = parameters.get_digit_parameters_None_if_absence(network_setting_ini, 'aspp_convs_filters','int')
    #
    #
    #     model_variant = parameters.get_string_parameters(network_setting_ini, 'model_variant')
    #
    #
    #     dataset = parameters.get_string_parameters(para_file, 'dataset_name')
    #     num_classes_noBG = parameters.get_digit_parameters_None_if_absence(para_file, 'NUM_CLASSES_noBG', 'int')
    #
    #     num_of_classes = num_classes_noBG + 1
    #     num_of_classes = 21 # for test
    #
    #     image_crop_size = parameters.get_string_list_parameters(para_file, 'image_crop_size')
    #     if len(image_crop_size) != 2 and image_crop_size[0].isdigit() and image_crop_size[1].isdigit():
    #         raise ValueError('image_crop_size should be height,width')
    #     crop_size_str = ','.join(image_crop_size)
    #
    #     evl_script = os.path.join(deeplab_dir, 'eval.py')
    #     evl_split = os.path.splitext(parameters.get_string_parameters(para_file, 'validation_sample_list_txt'))[0]
    #     max_eva_number = 1
    #
    #     # run evaluation
    #     deeplab_train.evaluation_deeplab(evl_script,dataset, evl_split, num_of_classes,model_variant,
    #                        inf_atrous_rates1,inf_atrous_rates2,inf_atrous_rates3,inf_output_stride,TRAIN_LOGDIR, EVAL_LOGDIR,
    #                        dataset_dir,crop_size_str, max_eva_number,depth_multiplier,decoder_output_stride,aspp_convs_filters)

    # this is easy to kill
    def calculation(self):
        a = 0
        while a < 1000:
            a += 1
            print(a)
            time.sleep(1)

    # start a sub-process, cannot end by kill or terminate
    # need to output the pid inside sub-prcocess, then red it and kill it.
    def run_a_subprocess(self):
        res = os.system('ping localhost')  # subprocess

    def test_Process(self):
        # eval_process = Process(target=self.calculation)
        eval_process = Process(target=self.run_a_subprocess)
        out_start = eval_process.start()
        print('out_start',out_start)
        print('pid',eval_process.pid)

        os_pid = os.getpid()
        print('os_pid', os_pid)
        pid = os.getpid()
        with open('train_py_pid.txt', 'w') as f_obj:
            f_obj.writelines('%d' % pid)

        with open('train_py_pid.txt', 'r') as f_obj:
            lines = f_obj.readlines()
            pid = int(lines[0].strip())
            print('read_pid', pid)

        time.sleep(5)
        eval_process.kill()
        # eval_process.terminate()
        time.sleep(3)
        print('is alive?',eval_process.is_alive())


if __name__ == '__main__':

    pass
