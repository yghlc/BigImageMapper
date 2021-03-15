#!/usr/bin/env python
# Filename: run_evaluation.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 13 March, 2021
"""

import os, sys
from optparse import OptionParser
import time

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)
import parameters
import basic_src.io_function as io_function

import utility.plot_miou_loss_curve as plot_miou_loss_curve

import workflow.deeplab_train as deeplab_train
from workflow.deeplab_train import evaluation_deeplab
from workflow.deeplab_train import pre_trained_tar_21_classes
from workflow.deeplab_train import pre_trained_tar_19_classes
from workflow.deeplab_train import get_miou_list_class_all

def run_evaluation(WORK_DIR, deeplab_dir, expr_name, para_file, network_setting_ini,gpu_num):

    EXP_FOLDER = expr_name
    TRAIN_LOGDIR = os.path.join(WORK_DIR,EXP_FOLDER,'train')
    EVAL_LOGDIR = os.path.join(WORK_DIR,EXP_FOLDER,'eval')
    dataset_dir = os.path.join(WORK_DIR, 'tfrecord')

    inf_output_stride = parameters.get_digit_parameters_None_if_absence(network_setting_ini,'inf_output_stride','int')
    inf_atrous_rates1 = parameters.get_digit_parameters_None_if_absence(network_setting_ini,'inf_atrous_rates1','int')
    inf_atrous_rates2 = parameters.get_digit_parameters_None_if_absence(network_setting_ini,'inf_atrous_rates2','int')
    inf_atrous_rates3 = parameters.get_digit_parameters_None_if_absence(network_setting_ini,'inf_atrous_rates3','int')

    b_initialize_last_layer = parameters.get_bool_parameters(para_file, 'b_initialize_last_layer')
    pre_trained_tar = parameters.get_string_parameters(network_setting_ini, 'TF_INIT_CKPT')

    # depth_multiplier default is 1.0.
    depth_multiplier = parameters.get_digit_parameters_None_if_absence(network_setting_ini, 'depth_multiplier', 'float')

    decoder_output_stride = parameters.get_digit_parameters_None_if_absence(network_setting_ini, 'decoder_output_stride', 'int')
    aspp_convs_filters = parameters.get_digit_parameters_None_if_absence(network_setting_ini, 'aspp_convs_filters','int')
    model_variant = parameters.get_string_parameters(network_setting_ini, 'model_variant')

    dataset = parameters.get_string_parameters(para_file,'dataset_name')
    num_classes_noBG = parameters.get_digit_parameters_None_if_absence(para_file, 'NUM_CLASSES_noBG', 'int')
    assert num_classes_noBG != None
    if b_initialize_last_layer is False:
        if pre_trained_tar in pre_trained_tar_21_classes:
            print('warning, pretrained model %s is trained with 21 classes, set num_of_classes to 21'%pre_trained_tar)
            num_classes_noBG = 20
        if pre_trained_tar in pre_trained_tar_19_classes:
            print('warning, pretrained model %s is trained with 19 classes, set num_of_classes to 19'%pre_trained_tar)
            num_classes_noBG = 18
    num_of_classes = num_classes_noBG + 1

    image_crop_size = parameters.get_string_list_parameters(para_file, 'image_crop_size')
    if len(image_crop_size) != 2 and image_crop_size[0].isdigit() and image_crop_size[1].isdigit():
        raise ValueError('image_crop_size should be height,width')
    crop_size_str = ','.join(image_crop_size)

    evl_script = os.path.join(deeplab_dir, 'eval.py')
    evl_split = os.path.splitext(parameters.get_string_parameters(para_file,'validation_sample_list_txt'))[0]
    max_eva_number = 1
    eval_interval_secs = 300

    # gpuid = ''      # do not use GPUs

    evaluation_deeplab(evl_script, dataset, evl_split, num_of_classes, model_variant,
                       inf_atrous_rates1, inf_atrous_rates2, inf_atrous_rates3, inf_output_stride, TRAIN_LOGDIR,
                       EVAL_LOGDIR,
                       dataset_dir, crop_size_str, max_eva_number, depth_multiplier, decoder_output_stride,
                       aspp_convs_filters,eval_interval_secs=eval_interval_secs)


    # get miou again
    miou_dict = get_miou_list_class_all(EVAL_LOGDIR, num_of_classes)

    # # get iou and backup
    # iou_path = os.path.join(EVAL_LOGDIR, 'miou.txt')
    #
    # # backup miou and training_loss & learning rate
    # test_id = os.path.basename(WORK_DIR) + '_' + expr_name
    # backup_dir = os.path.join(WORK_DIR, 'result_backup')
    # if os.path.isdir(backup_dir) is False:
    #     io_function.mkdir(backup_dir)
    # new_iou_name = os.path.join(backup_dir, test_id+ '_'+os.path.basename(iou_path))
    # io_function.copy_file_to_dst(iou_path, new_iou_name, overwrite=True)
    #
    # # plot mIOU, loss, and learnint rate curves, and backup
    # miou_curve_path = plot_miou_loss_curve.plot_miou_loss_main(iou_path)
    #
    # miou_curve_bakname = os.path.join(backup_dir, test_id+ '_'+os.path.basename(miou_curve_path))
    # io_function.copy_file_to_dst(miou_curve_path, miou_curve_bakname, overwrite=True)




def prepare_data_for_evaluation(para_file):

    import workflow.whole_procedure as whole_procedure

    # get subimages
    whole_procedure.extract_sub_images_using_training_polygons(para_file)

    # split image
    whole_procedure.split_sub_images(para_file)

    # covert image to tf-records
    whole_procedure.build_TFrecord_tf1x(para_file)


def main(options, args):

    print("%s : run evaluation" % os.path.basename(sys.argv[0]))
    SECONDS = time.time()

    para_file = args[0]
    gpu_num = 1
    b_new_validation_data = options.b_new_validation_data
    if os.path.isfile(para_file) is False:
        raise IOError('File %s not exists in current folder: %s' % (para_file, os.getcwd()))

    network_setting_ini = parameters.get_string_parameters(para_file, 'network_setting_ini')
    tf_research_dir = parameters.get_directory_None_if_absence(network_setting_ini, 'tf_research_dir')
    print(tf_research_dir)
    if tf_research_dir is None:
        raise ValueError('tf_research_dir is not in %s' % para_file)
    if os.path.isdir(tf_research_dir) is False:
        raise ValueError('%s does not exist' % tf_research_dir)
    # sys.path.insert(0, tf_research_dir)
    # sys.path.insert(0, os.path.join(tf_research_dir,'slim'))
    # print(sys.path)
    # need to change PYTHONPATH, otherwise, deeplab cannot be found
    if os.getenv('PYTHONPATH'):
        os.environ['PYTHONPATH'] = os.getenv('PYTHONPATH') + ':' + tf_research_dir + ':' + os.path.join(tf_research_dir,
                                                                                                        'slim')
    else:
        os.environ['PYTHONPATH'] = tf_research_dir + ':' + os.path.join(tf_research_dir, 'slim')
    # os.system('echo $PYTHONPATH ')

    tf1x_python = parameters.get_file_path_parameters(network_setting_ini,'tf1x_python')
    deeplab_train.tf1x_python = tf1x_python

    deeplab_dir = os.path.join(tf_research_dir, 'deeplab')
    WORK_DIR = os.getcwd()

    expr_name = parameters.get_string_parameters(para_file, 'expr_name')

    # prepare data for validation
    if b_new_validation_data:
        prepare_data_for_evaluation(para_file)


    run_evaluation(WORK_DIR, deeplab_dir, expr_name, para_file, network_setting_ini,gpu_num)

    duration= time.time() - SECONDS
    os.system('echo "$(date): time cost of running evaluation: %.2f seconds">>time_cost.txt'%duration)

if __name__ == '__main__':

    usage = "usage: %prog [options] para_file"
    parser = OptionParser(usage=usage, version="1.0 2021-03-13")
    parser.description = 'Introduction: run evaluation of Deeplab '

    parser.add_option("-n", "--b_new_validation_data",
                      action="store_true", dest="b_new_validation_data",default=False,
                      help="indicate whether indicate the validation is different with the one in folder, "
                           "need to create new image patches and tfrecord, also change some parameters in para file")

    (options, args) = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(2)

    main(options, args)
